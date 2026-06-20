from imwatermark import WatermarkDecoder
import cv2
import numpy as np
from PIL import Image

def detect_watermark(image_file):
    """
    Attempts to decode invisible watermarks (DWT/DCT) from the image.
    """
    try:
        # Load image for cv2
        image_file.seek(0)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Could not decode image for watermark detection"}

        # Attempt recovery of hidden watermark
        decoder = WatermarkDecoder('bytes', 32)
        
        watermark_payload = None
        confidence = 0.0
        
        try:
            watermark_payload = decoder.decode(img, 'dwtDct')
            if watermark_payload:
                try:
                    payload_str = watermark_payload.decode('utf-8')
                    if payload_str.isprintable():
                        confidence = 0.8
                except:
                    pass
        except Exception:
            pass
            
        return {
            "watermark_found": watermark_payload is not None and confidence > 0,
            "recovered_identifier": watermark_payload.decode('utf-8', errors='ignore') if watermark_payload else None,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}
