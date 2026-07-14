import os
import argparse
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from io import BytesIO
import time

# Mimir imports
from core.scoring import calculate_final_assessment
from detectors.ai_model import get_ai_detector
from detectors.metadata import extract_metadata
from detectors.watermark import detect_watermark
from forensics.ela import perform_ela

def simulate_whatsapp_compression(img_path):
    """
    Simulates WhatsApp's aggressive image compression:
    1. Strips all EXIF/metadata (by re-saving via PIL without exif).
    2. Resizes the image to a max dimension of 1600px.
    3. Saves it as a heavily compressed JPEG (Quality ~60).
    """
    img = Image.open(img_path)
    
    # Resize max dimension to 1600px
    max_dim = 1600
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    compressed_io = BytesIO()
    # Save as JPEG with 60% quality and NO exif data
    img.save(compressed_io, format='JPEG', quality=60)
    compressed_io.seek(0)
    return compressed_io

def run_benchmark(dataset_dir, simulate_whatsapp=True):
    print(f"Starting Mimir Benchmarking Tool")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"WhatsApp Compression Simulation: {'Enabled' if simulate_whatsapp else 'Disabled'}")
    print("-" * 50)
    
    y_true = []
    y_pred = []
    y_scores = []
    
    ai_detector = get_ai_detector()
    
    start_time = time.time()
    
    for category in ['real', 'fake']:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        for filename in os.listdir(cat_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            file_path = os.path.join(cat_dir, filename)
            is_fake_true = (category == 'fake')
            y_true.append(1 if is_fake_true else 0)
            
            # 1. Image preprocessing (Simulation)
            if simulate_whatsapp:
                file_obj = simulate_whatsapp_compression(file_path)
            else:
                file_obj = open(file_path, 'rb')
            
            # 2. Run Mimir Pipeline Layers
            # Layer 1: Metadata (Will be empty if compressed)
            metadata_results = extract_metadata(file_obj)
            
            # Layer 2: Watermark (Often destroyed by compression)
            watermark_results = detect_watermark(file_obj)
            
            # Layer 4: AI Visual Model (The workhorse post-compression)
            ai_results = ai_detector.predict(file_obj)
            
            # Layer 5: ELA (Noise analysis)
            forensic_results = perform_ela(file_obj)
            
            # Core Scoring
            score, label, conf = calculate_final_assessment(
                metadata_results, watermark_results, ai_results, forensic_results
            )
            
            y_scores.append(score)
            
            # Threshold for prediction (0.60 is the boundary for "Possible AI Generated")
            y_pred.append(1 if score > 0.60 else 0)
            
            if not simulate_whatsapp:
                file_obj.close()
                
    elapsed = time.time() - start_time
    
    if len(y_true) == 0:
        print("No images found in dataset directory. Expecting 'real' and 'fake' subfolders.")
        return
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # cm[0][0] = True Negative, cm[0][1] = False Positive
    # cm[1][0] = False Negative, cm[1][1] = True Positive
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n[BENCHMARK RESULTS]")
    print(f"Total Images Processed: {len(y_true)}")
    print(f"Time Taken: {elapsed:.2f} seconds")
    print(f"WhatsApp Metadata Stripping: {'Applied' if simulate_whatsapp else 'Not Applied'}")
    print("\nConfusion Matrix:")
    print(f"                Predicted Real(0)   Predicted Fake(1)")
    print(f"Actual Real(0)        {tn}                 {fp}")
    print(f"Actual Fake(1)        {fn}                 {tp}")
    
    print("\nMetrics:")
    print(f"Precision:            {precision:.4f} (When flagged as AI, how often is it actually AI?)")
    print(f"Recall (Sensitivity): {recall:.4f} (Out of all AI images, how many did we catch?)")
    print(f"False Positive Rate:  {fpr:.4f} (How often do we falsely accuse real images?)")
    print(f"F1-Score:             {f1_score(y_true, y_pred, zero_division=0):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mimir Benchmark and WhatsApp Simulation Tool")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset with 'real' and 'fake' subfolders")
    parser.add_argument("--no-compression", action="store_true", help="Disable WhatsApp compression simulation")
    args = parser.parse_args()
    
    run_benchmark(args.dataset, simulate_whatsapp=not args.no_compression)
