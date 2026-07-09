import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image

# TensorFlow is only needed for models using it
import tensorflow as tf  

# Local imports
from model_utils import preprocess_image, extract_features

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use non-GUI backend for matplotlib
matplotlib.use('Agg')

def is_skin_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        skin_ranges = [
            (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 150, 255], dtype=np.uint8)),
            (np.array([0, 20, 50], dtype=np.uint8), np.array([25, 150, 255], dtype=np.uint8)),
            (np.array([0, 20, 30], dtype=np.uint8), np.array([30, 150, 255], dtype=np.uint8))
        ]
        masks = [cv2.inRange(img_hsv, lower, upper) for lower, upper in skin_ranges]
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(masks[0], masks[1]), masks[2])

        skin_pixels = cv2.countNonZero(combined_mask)
        total_pixels = img.shape[0] * img.shape[1]
        skin_percentage = (skin_pixels / total_pixels) * 100

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_percentage = (cv2.countNonZero(edges) / total_pixels) * 100

        non_skin_mask = cv2.bitwise_not(combined_mask)
        non_skin_pixels = cv2.countNonZero(non_skin_mask)
        non_skin_avg_color = cv2.mean(img, mask=non_skin_mask)[:3] if non_skin_pixels > 0 else (0, 0, 0)

        logger.debug(f"Skin percentage: {skin_percentage:.2f}%")
        logger.debug(f"Edge percentage: {edge_percentage:.2f}%")
        logger.debug(f"Non-skin average color: {non_skin_avg_color}")

        return (
            skin_percentage > 20 and
            edge_percentage < 30 and
            sum(non_skin_avg_color) < 700
        )
    except Exception as e:
        logger.error(f"Error in skin detection: {e}")
        return False

def generate_gradcam(model, image_path, filename):
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return None

        logger.info(f"Input image path exists: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if img is None or img.size == 0:
            logger.error("Image load failed.")
            return None

        height, width = img.shape[:2]
        logger.debug(f"Image loaded with dimensions: {width}x{height}")

        # Preprocess
        processed_img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        # Get prediction
        try:
            prediction, confidence = model.predict(processed_img)
            logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

        # Generate activation map
        try:
            heatmap = model.get_activation_map(processed_img)
            if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
                logger.warning(f"Empty heatmap generated for {filename}, creating default heatmap")
                heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        except Exception as e:
            logger.error(f"Error generating activation map: {e}")
            heatmap = np.ones((224, 224), dtype=np.float32) * 0.5

        # Process heatmap
        heatmap = cv2.resize(heatmap, (width, height))
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        # Create visualization
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
        superimposed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

        # Add border based on prediction
        border_size = 10
        if prediction:  # Positive case
            border_color = (0, 0, 255)  # Red for positive
            text_color = (0, 0, 255)    # Red text
            result_text = "Leprosy Detected"
        else:  # Negative case
            border_color = (0, 255, 0)  # Green for negative
            text_color = (0, 255, 0)    # Green text
            result_text = "No Leprosy Detected"

        # Add border
        bordered_img = cv2.copyMakeBorder(
            superimposed_img, 
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Add text with background for better visibility
        text = f"{result_text}"
        confidence_text = f"Confidence: {confidence:.4f}"

        # Add semi-transparent background for text
        overlay = bordered_img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, bordered_img, 0.3, 0, bordered_img)

        # Add text
        cv2.putText(bordered_img, text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(bordered_img, confidence_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Save result
        output_filename = f"gradcam_{filename.rsplit('.', 1)[0]}.png"
        output_path = os.path.join('static', 'uploads', output_filename).replace('\\', '/')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if cv2.imwrite(output_path, bordered_img):
            logger.info(f"Grad-CAM image saved at: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to save Grad-CAM image at: {output_path}")
            return None

    except Exception as e:
        logger.error(f"Error generating Grad-CAM visualization: {e}")
        return None
