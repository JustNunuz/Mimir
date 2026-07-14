import os
import argparse
import sys

def download_dataset(output_dir, num_samples=1000):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' library is required to download from HuggingFace.")
        print("Please install it by running: pip install datasets")
        sys.exit(1)
        
    print(f"Downloading {num_samples} images from HuggingFace dataset...")
    # Using a common open-source dataset for AI vs Real images
    # We load the split 'train', and we will stream it so we don't download everything if it's huge
    dataset_name = "competitions/ai-generated-image-detection"
    
    try:
        # Load dataset in streaming mode to just pick 1000
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        # Fallback to another common dataset
        dataset_name = "artemiyf/ai-generated-images"
        print(f"Falling back to '{dataset_name}'...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    target_each = num_samples // 2
    
    print(f"Targeting {target_each} Real and {target_each} Fake images...")
    
    for sample in dataset:
        if real_count >= target_each and fake_count >= target_each:
            break
            
        # The column names depend on the dataset. We try common ones.
        image = sample.get("image") or sample.get("img")
        label = sample.get("label") or sample.get("is_ai")
        
        if image is None or label is None:
            continue
            
        # Standardize label (0 = real, 1 = fake/ai)
        # Note: adjust this logic if the specific dataset uses different label semantics
        is_fake = (label == 1 or str(label).lower() in ['fake', 'ai', '1', 'artificial'])
        
        if is_fake and fake_count < target_each:
            save_path = os.path.join(fake_dir, f"fake_{fake_count:04d}.jpg")
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(save_path, "JPEG")
            fake_count += 1
            if fake_count % 50 == 0:
                print(f"Downloaded {fake_count}/{target_each} Fake images")
                
        elif not is_fake and real_count < target_each:
            save_path = os.path.join(real_dir, f"real_{real_count:04d}.jpg")
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(save_path, "JPEG")
            real_count += 1
            if real_count % 50 == 0:
                print(f"Downloaded {real_count}/{target_each} Real images")

    print(f"Dataset download complete! Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download AI vs Real Image dataset from HuggingFace")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=1000, help="Total number of images to download")
    args = parser.parse_args()
    
    download_dataset(args.output, args.samples)