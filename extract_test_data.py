import os
import glob
import shutil
from sklearn.model_selection import train_test_split

def extract_test_data():
    dataset_path = os.path.join("data")
    image_paths = glob.glob(os.path.join(dataset_path, "images", "*.tif"))
    label_paths = glob.glob(os.path.join(dataset_path, "labels", "*.png"))
    
    if not image_paths:
        print(f"No .tif images found in {os.path.join(dataset_path, 'images')}.")
        print("Please ensure you have placed the raw dataset 'images' and 'labels' folders inside the 'data' directory.")
        return

    # To ensure consistent split, sort the paths first (notebook sorts them before making pairs)
    image_paths = sorted(image_paths)
    
    # Keep the exact same random_state=42 and test_size=0.15 used during training
    train_imgs, test_imgs = train_test_split(image_paths, test_size=0.15, random_state=42)

    output_dir = "test_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(test_imgs)} test images. Copying them to '{output_dir}'...")
    
    count = 0
    for img_path in test_imgs:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(img_path, dest_path)
        count += 1
        
    print(f"Successfully isolated {count} test images across the exact same split.")

if __name__ == "__main__":
    extract_test_data()
