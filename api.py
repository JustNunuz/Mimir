from fastapi import FastAPI, File, UploadFile
import io
import uvicorn

from detectors.metadata import analyze_metadata
from detectors.watermark import detect_watermark
from detectors.hashing import generate_hashes
from detectors.ai_model import get_ai_detector
from forensics.ela import error_level_analysis
from core.scoring import calculate_final_assessment
from registry.db import RegistryDB

app = FastAPI(title="Mimir API", description="AI Image Provenance Detection API")
db = RegistryDB()
detector = get_ai_detector()

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    contents = await file.read()
    file_obj = io.BytesIO(contents)
    
    metadata_res = analyze_metadata(file_obj)
    watermark_res = detect_watermark(file_obj)
    hash_res = generate_hashes(file_obj)
    ai_res = detector.predict(file_obj)
    ela_res = error_level_analysis(file_obj)
    
    final_score, label, conf = calculate_final_assessment(
        metadata_res, watermark_res, ai_res, ela_res
    )
    
    record_id = db.add_record(
        filename=file.filename,
        phash=hash_res.get("phash"),
        dhash=hash_res.get("dhash"),
        ahash=hash_res.get("ahash"),
        watermark_id=watermark_res.get("recovered_identifier"),
        metadata=metadata_res.get("findings"),
        ai_score=ai_res.get("ai_probability", 0),
        human_score=ai_res.get("human_probability", 0),
        final_assessment=label
    )
    
    return {
        "record_id": record_id,
        "assessment": label,
        "ai_probability": final_score,
        "confidence": conf,
        "layers": {
            "metadata": metadata_res,
            "watermark": watermark_res,
            "hashing": hash_res,
            "ai_model": ai_res,
            "forensics": ela_res
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
