import exifread
import piexif
from PIL import Image

def analyze_metadata(image_file):
    """
    Extracts metadata and looks for signatures of AI generation tools.
    """
    try:
        image_file.seek(0)
        tags = exifread.process_file(image_file)
    except Exception as e:
        tags = {}
    
    findings = []
    metadata_dict = {}
    
    ai_keywords = [
        "openai", "dall-e", "midjourney", "stable diffusion", 
        "adobe firefly", "gemini", "flux", "content credentials", 
        "c2pa", "trainedalgorithmicmedia"
    ]
    
    # 1. Standard EXIF Check
    for tag, value in tags.items():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            val_str = str(value).lower()
            metadata_dict[tag] = str(value)
            
            for keyword in ai_keywords:
                if keyword in val_str:
                    findings.append({
                        "source": f"EXIF: {tag}",
                        "keyword_found": keyword,
                        "raw_value": str(value)
                    })
                    
    # 2. Raw XMP / C2PA Search
    try:
        image_file.seek(0)
        raw_bytes = image_file.read()
        raw_str = raw_bytes.decode('latin-1').lower()
        
        for keyword in ai_keywords:
            if keyword in raw_str:
                # Avoid duplicating findings from EXIF
                if not any(f["keyword_found"] == keyword for f in findings):
                    findings.append({
                        "source": "Raw XMP/C2PA Data",
                        "keyword_found": keyword,
                        "raw_value": f"Binary substring match: {keyword}"
                    })
    except Exception:
        pass
                    
    # Determine confidence based on findings
    evidence_score = 0.0
    if len(findings) > 0:
        evidence_score = 0.9 # High confidence if explicit AI keywords found in metadata
        
    return {
        "metadata_keys_count": len(metadata_dict),
        "findings": findings,
        "evidence_score": evidence_score,
        "raw_metadata": metadata_dict
    }
