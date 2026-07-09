#!/usr/bin/env python3
"""
Test the model specifically on leprosy positive images from the dataset.
"""

import os
import logging
import numpy as np
import cv2
from PIL import Image
from model_utils import load_model, preprocess_image
from xai_utils import generate_gradcam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_border_and_text(img, prediction, confidence):
    """Add colored border and text to image based on prediction"""
    # Add border
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
        img, 
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
    
    return bordered_img

def test_leprosy_positive():
    """Test the model on leprosy positive images"""
    # Load model
    model = load_model('model/leprosy_classifier.keras')
    if not model:
        logger.error("Failed to load model")
        return
    
    logger.info("Model loaded successfully")
    
    # Path to positive samples
    pos_samples_dir = 'dataset/leprosy_dataset/positive'
    if not os.path.exists(pos_samples_dir):
        logger.error(f"Positive samples directory not found: {pos_samples_dir}")
        return
    
    # Create output directory for test results
    output_dir = 'test_results/leprosy_positive'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each positive image
    for filename in os.listdir(pos_samples_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(pos_samples_dir, filename)
            logger.info(f"\nTesting image: {filename}")
            
            try:
                # Load and preprocess image
                img = preprocess_image(image_path)
                if img is None:
                    logger.error(f"Failed to load image: {image_path}")
                    continue
                
                # Get prediction
                prediction, confidence = model.predict(img)
                
                # Log results
                logger.info(f"Prediction: {'Positive' if prediction else 'Negative'}")
                logger.info(f"Confidence: {confidence:.4f}")
                
                # Generate Grad-CAM visualization
                gradcam_path = generate_gradcam(model, image_path, filename)
                if gradcam_path:
                    logger.info(f"Grad-CAM visualization saved to: {gradcam_path}")
                
                # Save original image with prediction and border
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    # Add border and text
                    result_img = add_border_and_text(original_img, prediction, confidence)
                    
                    # Save annotated image
                    output_path = os.path.join(output_dir, f"result_{filename}")
                    cv2.imwrite(output_path, result_img)
                    logger.info(f"Annotated image saved to: {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                continue

if __name__ == "__main__":
    test_leprosy_positive() 