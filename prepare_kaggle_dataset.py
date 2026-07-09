import os
import kaggle
import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset paths
DATASET_BASE_DIR = "dataset/leprosy_dataset"
POSITIVE_DIR = os.path.join(DATASET_BASE_DIR, "positive")
NEGATIVE_DIR = os.path.join(DATASET_BASE_DIR, "negative")

# Kaggle datasets to try (in order of preference)
KAGGLE_DATASETS = [
    "aryashah2k/skin-disease-dataset",  # General skin disease dataset
    "nikhilroxtomar/leprosy-detection-using-deep-learning",  # Specific leprosy dataset
    "shubhamgoel27/dermnet",  # Dermatology dataset
]

def setup_directories():
    """Create necessary directories for the dataset"""
    os.makedirs(POSITIVE_DIR, exist_ok=True)
    os.makedirs(NEGATIVE_DIR, exist_ok=True)
    logger.info("Created dataset directories")

def download_kaggle_dataset():
    """Download dataset from Kaggle"""
    try:
        # Try each dataset until one works
        for dataset in KAGGLE_DATASETS:
            try:
                logger.info(f"Attempting to download dataset: {dataset}")
                kaggle.api.dataset_download_files(dataset, path="temp_dataset", unzip=True)
                logger.info(f"Successfully downloaded dataset: {dataset}")
                return "temp_dataset"
            except Exception as e:
                logger.warning(f"Failed to download {dataset}: {e}")
                continue
        
        logger.error("Failed to download any dataset")
        return None
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def process_dataset(dataset_path):
    """Process the downloaded dataset and organize it"""
    try:
        # Keywords to identify leprosy-related images
        leprosy_keywords = [
            'leprosy', 'hansen', 'mycobacterium leprae',
            'leprae', 'lepra', 'hansen\'s disease'
        ]
        
        # Process all images in the dataset
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(root, file)
                    
                    # Check if file name or path contains leprosy keywords
                    is_leprosy = any(keyword in file.lower() or keyword in root.lower() 
                                   for keyword in leprosy_keywords)
                    
                    # Copy to appropriate directory
                    if is_leprosy:
                        shutil.copy2(file_path, os.path.join(POSITIVE_DIR, file))
                        logger.info(f"Copied leprosy image: {file}")
                    else:
                        # For non-leprosy images, check if it's a skin image
                        # You might want to add more sophisticated checks here
                        shutil.copy2(file_path, os.path.join(NEGATIVE_DIR, file))
                        logger.info(f"Copied non-leprosy image: {file}")
        
        # Print summary
        positive_count = len(os.listdir(POSITIVE_DIR))
        negative_count = len(os.listdir(NEGATIVE_DIR))
        
        logger.info(f"\nDataset processing complete!")
        logger.info(f"Positive samples: {positive_count}")
        logger.info(f"Negative samples: {negative_count}")
        logger.info(f"Total samples: {positive_count + negative_count}")
        
        return DATASET_BASE_DIR
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return None

def main():
    """Main function to download and prepare the dataset"""
    print("Starting Kaggle dataset preparation...")
    
    # Create directories
    setup_directories()
    
    # Download dataset
    dataset_path = download_kaggle_dataset()
    if not dataset_path:
        print("Failed to download dataset. Please check your Kaggle credentials and internet connection.")
        return
    
    # Process dataset
    final_path = process_dataset(dataset_path)
    if final_path:
        print(f"\nDataset has been prepared in: {final_path}")
        print("You can now train the model using: python train_model.py")
    
    # Clean up temporary files
    if os.path.exists("temp_dataset"):
        shutil.rmtree("temp_dataset")

if __name__ == "__main__":
    main() 