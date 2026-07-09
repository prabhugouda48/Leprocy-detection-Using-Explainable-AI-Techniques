import os
import numpy as np
import logging
import pickle
import time
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from flask import after_this_request

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnhancedLeproyModel:
    """
    An advanced CNN-based model for leprosy detection using EfficientNet
    """
    def __init__(self, model_path='model/leprosy_classifier.keras'):
        self.name = "LeprosiNet"
        self.model_path = model_path
        self.model = None
        self.target_size = (224, 224)
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                # Log model structure
                logger.info("Model structure:")
                for layer in self.model.layers:
                    logger.info(f"Layer: {layer.name}, Type: {type(layer).__name__}")
            else:
                logger.info("Creating new model")
                self.model = self._create_new_model()
        except Exception as e:
            logger.error(f"Error loading/creating model: {e}")
            self.model = self._create_new_model()
    
    def _create_new_model(self):
        """Create a new CNN model with enhanced leprosy detection capabilities"""
        # Load base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.target_size, 3)
        )
        
        # Unfreeze some layers for fine-tuning
        for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
            layer.trainable = True
        
        # Add custom layers with more capacity
        x = base_model.output
        x = GlobalAveragePooling2D(name='gap_layer')(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid', name='prediction_layer')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model with adjusted learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, train_data, epochs=50, batch_size=32):
        """Train the model with enhanced data augmentation"""
        try:
            # Create data generators with stronger augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,  # Increased rotation
                width_shift_range=0.4,  # Increased shift
                height_shift_range=0.4,
                shear_range=0.4,
                zoom_range=0.4,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                validation_split=0.2,
                brightness_range=[0.8, 1.2],  # Added brightness variation
                channel_shift_range=50.0,  # Added color channel variation
                preprocessing_function=lambda x: x + np.random.normal(0, 0.1, x.shape)  # Added noise
            )
            
            # Load training data
            train_generator = train_datagen.flow_from_directory(
                train_data,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode='binary',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                train_data,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode='binary',
                subset='validation'
            )
            
            # Calculate class weights to handle imbalance
            total_samples = len(train_generator.filenames)
            positive_samples = sum(1 for f in train_generator.filenames if 'positive' in f)
            negative_samples = total_samples - positive_samples
            
            # Adjust class weights to give more importance to positive cases
            class_weights = {
                0: total_samples / (2 * negative_samples),  # Weight for negative class
                1: total_samples / (1.5 * positive_samples)  # Increased weight for positive class
            }
            
            # Train the model with adjusted parameters
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=5,
                        min_lr=1e-6
                    )
                ]
            )
            
            # Save the trained model
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None
    
    def predict(self, image):
        """
        Make prediction on preprocessed image with adjusted sensitivity for leprosy detection
        
        Args:
            image: Preprocessed image array
            
        Returns:
            tuple: (binary_prediction, confidence_value)
        """
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(image, verbose=0)
            
            # Handle different prediction output formats
            if isinstance(prediction, np.ndarray):
                if prediction.ndim > 1:
                    pred_value = float(prediction[0][0])
                else:
                    pred_value = float(prediction[0])
            elif isinstance(prediction, tf.Tensor):
                pred_value = float(prediction.numpy()[0])
            else:
                pred_value = float(prediction)
                
            logger.debug(f"Raw prediction value: {pred_value}")
            
            # Ensure we have a valid prediction value
            if np.isnan(pred_value) or np.isinf(pred_value):
                logger.warning("Invalid prediction value detected")
                return False, 0.0
            
            # Convert to binary prediction with adjusted threshold for better sensitivity
            # Lower threshold to catch more potential leprosy cases
            binary_prediction = pred_value > 0.25  # Lowered from 0.3 to 0.25
            
            # Return both the binary prediction and the raw confidence value
            return bool(binary_prediction), float(pred_value)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return False, 0.0
    
    def get_activation_map(self, image):
        """
        Generate Grad-CAM activation map for the input image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            numpy.ndarray: Activation map
        """
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                logger.warning("No convolutional layer found for Grad-CAM")
                return np.zeros((224, 224))
            
            # Create a model that maps the input image to the activations
            # of the last conv layer and the output predictions
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            
            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, 0]  # Binary classification
            
            # Extract gradients and compute guided gradients
            output = conv_outputs[0]
            grads = tape.gradient(loss, conv_outputs)[0]
            
            # Global average pooling of gradients
            weights = tf.reduce_mean(grads, axis=(0, 1))
            
            # Weight the channels by corresponding gradient
            cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
            
            # Apply ReLU and normalize
            cam = tf.maximum(cam, 0)
            cam = cam / tf.reduce_max(cam)
            
            # Resize to input image size
            cam = tf.image.resize(cam[..., tf.newaxis], self.target_size)
            cam = tf.squeeze(cam)
            
            return cam.numpy()
            
        except Exception as e:
            logger.error(f"Error generating activation map: {e}")
            return np.zeros(self.target_size)

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Enhanced image preprocessing for medical images
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_image(model, img_array):
    """
    Make prediction on preprocessed image with adjusted sensitivity
    
    Args:
        model: Loaded model
        img_array: Preprocessed image array
        
    Returns:
        tuple: (prediction, confidence)
    """
    try:
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction = model.predict(img_array)[0][0]
        
        # Lower the threshold for positive detection to increase sensitivity
        # This means we'll be more likely to detect leprosy
        binary_prediction = prediction > 0.3  # Lowered from 0.5 to 0.3
        
        # Get confidence score
        confidence = float(prediction if binary_prediction else 1 - prediction)
        
        logger.info(f"Raw prediction score: {prediction:.4f}")
        logger.info(f"Final prediction: {binary_prediction}, Confidence: {confidence:.4f}")
        return binary_prediction, confidence
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return False, 0.0

def load_model(model_path='model/leprosy_classifier.keras'):
    """
    Load the leprosy detection model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        model: The loaded model
    """
    try:
        return EnhancedLeproyModel(model_path)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        # Return a new model instance
        return EnhancedLeproyModel()

def create_model():
    """
    Create a new leprosy detection model
    
    Returns:
        model: A new model instance
    """
    try:
        model = EnhancedLeproyModel()
        model._create_new_model()
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return EnhancedLeproyModel()

def extract_features(img_array):
    """
    Extract features from preprocessed image
    
    Args:
        img_array: Preprocessed image array
        
    Returns:
        features: Extracted features for model input
    """
    try:
        # Flatten the image if it's multidimensional
        if len(img_array.shape) > 2:
            # For RGB images, use all channels
            if len(img_array.shape) == 3:
                # Extract color features
                avg_color_per_channel = np.mean(img_array, axis=(0, 1))
                std_color_per_channel = np.std(img_array, axis=(0, 1))
                
                # Convert to grayscale for texture features
                if img_array.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(
                        (img_array * 255).astype(np.uint8), 
                        cv2.COLOR_RGB2GRAY
                    )
                else:
                    gray = (img_array[:, :, 0] * 255).astype(np.uint8)
            else:
                # Handle batch dimension if present
                first_img = img_array[0]
                avg_color_per_channel = np.mean(first_img, axis=(0, 1))
                std_color_per_channel = np.std(first_img, axis=(0, 1))
                
                if first_img.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(
                        (first_img * 255).astype(np.uint8), 
                        cv2.COLOR_RGB2GRAY
                    )
                else:
                    gray = (first_img[:, :, 0] * 255).astype(np.uint8)
                
            # Basic histogram features
            hist_features = []
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            hist_features.extend(hist)
            
            # Edge detection features
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Simple texture features
            texture_features = [
                np.std(gray),     # Standard deviation
                np.mean(gray),    # Mean intensity
                edge_density      # Edge density
            ]
            
            # Combine all features
            features = np.concatenate([
                avg_color_per_channel, 
                std_color_per_channel,
                hist_features,
                texture_features
            ])
            
            # Reshape for sklearn (samples, features)
            return features.reshape(1, -1)
        else:
            # If already flattened, just return it
            return img_array.reshape(1, -1)
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        # Return consistent feature vector
        feature_size = 3 + 3 + 16 + 3  # RGB means + RGB stds + histogram bins + texture features
        return np.zeros((1, feature_size))

def train_model_from_dataset(dataset_dir, output_path='model/leprosy_classifier.keras'):
    try:
        model = EnhancedLeproyModel(output_path)
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset directory {dataset_dir} not found.")
            return model

        # Directly train using the CNN training method
        history = model.train(dataset_dir)
        logger.info("Model training completed successfully!")
        return model

    except Exception as e:
        logger.error(f"Error training model from dataset: {e}")
        return EnhancedLeproyModel(output_path)

        
        # Look for class folders (positive and negative)
        positive_dir = os.path.join(dataset_dir, 'positive')
        negative_dir = os.path.join(dataset_dir, 'negative')
        
        # Check if these directories exist
        if not all(os.path.exists(d) for d in [positive_dir, negative_dir]):
            logger.error(f"Missing class directories in {dataset_dir}")
            return model
        
        # Load and preprocess images from both classes
        positive_images = os.listdir(positive_dir)
        negative_images = os.listdir(negative_dir)
        
        # Create features and labels arrays
        all_features = []
        all_labels = []
        
        # Process positive samples
        logger.info(f"Processing {len(positive_images)} positive samples...")
        for img_file in positive_images:
            img_path = os.path.join(positive_dir, img_file)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                all_features.append(features[0])
                all_labels.append(1)  # Positive class
        
        # Process negative samples
        logger.info(f"Processing {len(negative_images)} negative samples...")
        for img_file in negative_images:
            img_path = os.path.join(negative_dir, img_file)
            if os.path.isfile(img_path):
                img = preprocess_image(img_path)
                features = extract_features(img)
                all_features.append(features[0])
                all_labels.append(0)  # Negative class
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Train the model
        logger.info(f"Training model with {len(X)} samples...")
        if len(X) > 0:
            model.train(X, y)
            logger.info("Model training completed successfully!")
        else:
            logger.error("No valid samples found for training.")
        
        return model
    except Exception as e:
        logger.error(f"Error training model from dataset: {e}")
        return EnhancedLeproyModel(output_path)

