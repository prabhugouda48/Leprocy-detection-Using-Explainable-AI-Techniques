#!/usr/bin/env python3
"""
Test the trained leprosy detection model with a few sample images.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from model_utils import load_model, preprocess_image, predict_image
from xai_utils import generate_gradcam

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_on_samples():
    """Test the model on sample images from each category"""
    model = load_model('model/leprosy_classifier.keras')
    
    if not model:
        logger.error("Failed to load model")
        return
    
    logger.info("Model loaded successfully")
    
    # Test positive sample
    pos_samples_dir = 'dataset/leprosy_dataset/positive'
    if os.path.exists(pos_samples_dir):
        pos_samples = os.listdir(pos_samples_dir)
        if pos_samples:
            logger.info(f"Testing positive sample: {pos_samples[0]}")
            test_sample(model, os.path.join(pos_samples_dir, pos_samples[0]), expected=True)
    
    # Test negative sample
    neg_samples_dir = 'dataset/leprosy_dataset/negative'
    if os.path.exists(neg_samples_dir):
        neg_samples = os.listdir(neg_samples_dir)
        if neg_samples:
            logger.info(f"Testing negative sample: {neg_samples[0]}")
            test_sample(model, os.path.join(neg_samples_dir, neg_samples[0]), expected=False)
    
def test_sample(model, image_path, expected):
    """Test a single sample with detailed logging"""
    try:
        # Load and preprocess image
        img = preprocess_image(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Get prediction
        prediction, confidence = model.predict(img)
        
        # Log detailed information
        logger.info(f"Image: {os.path.basename(image_path)}")
        logger.info(f"Expected: {'Positive' if expected else 'Negative'}")
        logger.info(f"Predicted: {'Positive' if prediction else 'Negative'}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        # Generate Grad-CAM visualization
        gradcam_path = generate_gradcam(model, image_path, os.path.basename(image_path))
        if gradcam_path:
            logger.info(f"Grad-CAM visualization saved to: {gradcam_path}")
        
        # Check if prediction matches expectation
            if prediction == expected:
                logger.info("✓ Prediction matches expectation")
            else:
                logger.warning("✗ Prediction does not match expectation")
        
    except Exception as e:
        logger.error(f"Error testing sample: {e}")

if __name__ == "__main__":
    test_model_on_samples()