import os
import shutil
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_dataset(source_dir, target_dir='"C:\Users\haris\FINAL YEAR PROJECT\archive\CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients"'):
    """
    Organize your custom dataset into the required structure
    
    Args:
        source_dir (str): Path to your source dataset directory
        target_dir (str): Path where the organized dataset will be saved
    """
    try:
        # Create target directories
        positive_dir = os.path.join(target_dir, 'positive')
        negative_dir = os.path.join(target_dir, 'negative')
        
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)
        
        # Get all image files from source directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        all_files = []
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    all_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(all_files)} images in source directory")
        
        # Ask user to classify each image
        for img_path in all_files:
            print(f"\nImage: {img_path}")
            print("Is this image showing leprosy? (y/n/q to quit)")
            response = input().lower()
            
            if response == 'q':
                break
            elif response == 'y':
                # Copy to positive directory
                shutil.copy2(img_path, os.path.join(positive_dir, os.path.basename(img_path)))
                logger.info(f"Copied {img_path} to positive directory")
            elif response == 'n':
                # Copy to negative directory
                shutil.copy2(img_path, os.path.join(negative_dir, os.path.basename(img_path)))
                logger.info(f"Copied {img_path} to negative directory")
        
        # Print summary
        positive_count = len(os.listdir(positive_dir))
        negative_count = len(os.listdir(negative_dir))
        
        logger.info(f"\nDataset organization complete!")
        logger.info(f"Positive samples: {positive_count}")
        logger.info(f"Negative samples: {negative_count}")
        logger.info(f"Total samples: {positive_count + negative_count}")
        
        return target_dir
        
    except Exception as e:
        logger.error(f"Error organizing dataset: {e}")
        return None

if __name__ == "__main__":
    print("Welcome to the Dataset Organizer!")
    print("This script will help you organize your custom dataset for leprosy detection.")
    print("\nPlease enter the path to your source dataset directory:")
    source_dir = input().strip()
    
    if os.path.exists(source_dir):
        target_dir = organize_dataset(source_dir)
        if target_dir:
            print(f"\nYour dataset has been organized in: {target_dir}")
            print("You can now train the model using: python train_model.py")
    else:
        print(f"Error: Directory '{source_dir}' does not exist!") 