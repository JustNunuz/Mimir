import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import base64

def error_level_analysis(image_file, quality=90):
    """
    Performs Error Level Analysis (ELA) to detect tampering or compression artifacts.
    """
    try:
        image_file.seek(0)
        original = Image.open(image_file).convert('RGB')
        
        # Save at given quality
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, 'JPEG', quality=quality)
        temp_buffer.seek(0)
        compressed = Image.open(temp_buffer)
        
        # Calculate difference
        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Save ELA image to bytes for display
        ela_buffer = io.BytesIO()
        ela_image.save(ela_buffer, format="JPEG")
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')
        
        # Generate a signal score based on the variance of the ELA image
        ela_gray = ela_image.convert('L')
        ela_array = np.array(ela_gray)
        variance = np.var(ela_array)
        
        # Local contrast/anomaly heuristic instead of raw variance
        median_val = np.median(ela_array)
        p95_val = np.percentile(ela_array, 95)
        local_anomaly = float(p95_val - median_val) / 255.0
        
        # Calculate anomaly score (0 to 1)
        anomaly_score = min(1.0, local_anomaly * 1.5) 
        
        return {
            "variance": float(variance),
            "anomaly_score": float(anomaly_score),
            "ela_max_diff": float(max_diff),
            "ela_image_base64": ela_base64
        }
        
    except Exception as e:
        return {"error": str(e)}
