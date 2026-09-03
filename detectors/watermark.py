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
        
        confidence = 0.0
        dwtdct_ok = False
        dwtdctsvd_ok = False
        payload_str_dct = None
        payload_str_svd = None
        
        try:
            watermark_payload_dct = decoder.decode(img, 'dwtDct')
            if watermark_payload_dct:
                payload_str_dct = watermark_payload_dct.decode('utf-8')
                if payload_str_dct.isprintable():
                    dwtdct_ok = True
                    confidence = 0.8
        except Exception:
            pass
            
        try:
            watermark_payload_svd = decoder.decode(img, 'dwtDctSvd')
            if watermark_payload_svd:
                payload_str_svd = watermark_payload_svd.decode('utf-8')
                if payload_str_svd.isprintable():
                    dwtdctsvd_ok = True
                    confidence = max(confidence, 0.8)
        except Exception:
            pass
            
        cross_method_agreement = bool(dwtdct_ok and dwtdctsvd_ok)
        
        return {
            "watermark_found": cross_method_agreement,
            "soft_signal": confidence,
            "recovered_identifier": payload_str_dct if payload_str_dct else payload_str_svd,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}
