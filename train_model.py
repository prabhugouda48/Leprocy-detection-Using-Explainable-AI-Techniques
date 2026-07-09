import os
import logging
import numpy as np
from model_utils import EnhancedLeproyModel, preprocess_image
from xai_utils import is_skin_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_leprosy_model(data_dir='dataset/leprosy_dataset', 
                        output_path='model/leprosy_classifier.keras'):
    """
    Train a CNN model for leprosy detection using your custom dataset
    """
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found.")
            return None
        
        # Check for positive and negative directories
        positive_dir = os.path.join(data_dir, 'positive')
        negative_dir = os.path.join(data_dir, 'negative')
        
        if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
            logger.error(f"Missing positive or negative directories in {data_dir}")
            return None
        
        # Count images in each category
        positive_count = len([f for f in os.listdir(positive_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        negative_count = len([f for f in os.listdir(negative_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        logger.info(f"Found {positive_count} positive samples and {negative_count} negative samples")
        
        if positive_count == 0 or negative_count == 0:
            logger.error("No images found in one or both categories")
            return None
        
        # Create and train model
        logger.info("Creating model...")
        model = EnhancedLeproyModel(output_path)
        
        logger.info("Starting model training...")
        history = model.train(
            train_data=data_dir,
            epochs=50,  # You can adjust this number
            batch_size=32  # You can adjust this number
        )
        
        logger.info("Model training completed!")
        return history
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None

if __name__ == "__main__":
    dataset_path = "dataset/leprosy_dataset"
    
    print("Starting model training with your custom dataset...")
    history = train_leprosy_model(
        data_dir=dataset_path,
        output_path='model/leprosy_classifier.keras'
    )
    
    if history:
        print("\nTraining completed successfully!")
        print("The model has been saved as: model/leprosy_classifier.keras")
    else:
        print("\nTraining failed. Please check the error messages above.")
