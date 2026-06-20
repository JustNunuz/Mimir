import imagehash
from PIL import Image

def generate_hashes(image_file):
    """
    Generates perceptual hashes for robust fingerprinting.
    """
    try:
        image_file.seek(0)
        img = Image.open(image_file)
        
        phash = str(imagehash.phash(img))
        dhash = str(imagehash.dhash(img))
        ahash = str(imagehash.average_hash(img))
        
        return {
            "phash": phash,
            "dhash": dhash,
            "ahash": ahash
        }
    except Exception as e:
        return {"error": str(e)}
