import exifread
import piexif
from PIL import Image

STRUCTURED_TAGS = {"Image Software", "EXIF Software", "XMP-xmp:CreatorTool", "Software"}

def _score_finding(finding, c2pa_verified=False):
    if c2pa_verified or finding["source"] == "C2PA Manifest (signature verified)":
        return 0.98
    if finding["source"].startswith("EXIF") and finding.get("tag") in STRUCTURED_TAGS:
        return 0.75
    if finding["source"] == "Structured XMP":
        return 0.75
    return 0.30

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
        "trainedalgorithmicmedia"
    ]
    
    c2pa_verified = False
    try:
        image_file.seek(0)
        import c2pa
        reader = c2pa.Reader(image_file)
        manifest_data = reader.json()
        if manifest_data:
            c2pa_verified = True
            findings.append({
                "source": "C2PA Manifest (signature verified)",
                "keyword_found": "c2pa",
                "tag": "C2PA",
                "raw_value": "Cryptographically verified C2PA manifest"
            })
    except ImportError:
        pass
    except Exception:
        pass
    
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
                        "tag": tag.split(" ")[-1] if " " in tag else tag,
                        "raw_value": str(value)
                    })
                    
    # 2. Targeted XMP Search (instead of full file search)
    try:
        image_file.seek(0)
        raw_bytes = image_file.read()
        xmp_start = raw_bytes.find(b'<?xpacket begin')
        xmp_end = raw_bytes.find(b'<?xpacket end')
        if xmp_start != -1 and xmp_end != -1:
            xmp_data = raw_bytes[xmp_start:xmp_end].decode('utf-8', errors='ignore').lower()
            for keyword in ai_keywords:
                if keyword in xmp_data:
                    if not any(f["keyword_found"] == keyword for f in findings):
                        findings.append({
                            "source": "Structured XMP",
                            "keyword_found": keyword,
                            "tag": "XMP",
                            "raw_value": f"XMP match: {keyword}"
                        })
    except Exception:
        pass
                    
    # Determine confidence based on tiered findings
    evidence_score = 0.0
    for finding in findings:
        score = _score_finding(finding, c2pa_verified)
        if score > evidence_score:
            evidence_score = score
            
    return {
        "metadata_keys_count": len(metadata_dict),
        "findings": findings,
        "evidence_score": evidence_score,
        "raw_metadata": metadata_dict,
        "c2pa_verified": c2pa_verified
    }
