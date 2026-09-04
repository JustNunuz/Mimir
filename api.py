import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile
import io
import uvicorn

from detectors.metadata import analyze_metadata
from detectors.watermark import detect_watermark
from detectors.hashing import generate_hashes
from detectors.ai_model import get_ai_detector
from forensics.ela import combined_forensic_analysis
from core.scoring import calculate_final_assessment
from registry.db import RegistryDB

logger = logging.getLogger(__name__)

app = FastAPI(title="Mimir API", description="AI Image Provenance Detection API")
db = RegistryDB()
detector = get_ai_detector()

@app.post("/scan")
async def scan_image(
    file: UploadFile = File(...),
    include_images: bool = Query(
        default=False,
        description="Include base64-encoded ELA and spectrum images in the response. "
                    "Disabled by default — callers that only need the score/label "
                    "don't pay the bandwidth cost."
    )
):
    contents = await file.read()

    # Offload blocking, CPU-bound tasks to threadpool concurrently.
    # return_exceptions=True: a single detector failure returns an Exception object
    # rather than cancelling the whole gather — survivors degrade gracefully.
    results = await asyncio.gather(
        asyncio.to_thread(analyze_metadata, io.BytesIO(contents)),
        asyncio.to_thread(detect_watermark, io.BytesIO(contents)),
        asyncio.to_thread(generate_hashes, io.BytesIO(contents)),
        asyncio.to_thread(detector.predict, io.BytesIO(contents)),
        asyncio.to_thread(combined_forensic_analysis, io.BytesIO(contents)),
        return_exceptions=True,
    )

    # Unpack; convert any raised exceptions to the same {"error": ...} shape the
    # detectors already use for internal failures, so downstream code stays uniform.
    def _safe(result, name: str) -> dict:
        if isinstance(result, Exception):
            logger.error("Detector '%s' raised: %s", name, result, exc_info=result)
            return {"error": str(result)}
        return result

    metadata_res  = _safe(results[0], "metadata")
    watermark_res = _safe(results[1], "watermark")
    hash_res      = _safe(results[2], "hashing")
    ai_res        = _safe(results[3], "ai_model")
    forensic_res  = _safe(results[4], "forensics")

    final_score, label, conf = calculate_final_assessment(
        metadata_res, watermark_res, ai_res, forensic_res
    )

    record_id = await asyncio.to_thread(
        db.add_record,
        filename=file.filename,
        phash=hash_res.get("phash"),
        dhash=hash_res.get("dhash"),
        ahash=hash_res.get("ahash"),
        watermark_id=watermark_res.get("recovered_identifier"),
        metadata=metadata_res.get("findings"),
        # Store the post-RL-blend, post-floor score the user actually saw, not the raw
        # ensemble output — so the registry matches the displayed result.
        ai_score=final_score,
        human_score=1.0 - final_score,
        final_assessment=label
    )

    # Strip base64 image blobs by default — bots only need the score/label/confidence.
    # Pass ?include_images=true to get the visualizations.
    if not include_images:
        forensic_res = {
            k: (
                {ik: iv for ik, iv in v.items()
                 if not ik.endswith("_base64")} if isinstance(v, dict) else v
            )
            for k, v in forensic_res.items()
        }
        ai_res = {k: v for k, v in ai_res.items() if not k.endswith("_base64")}

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
            "forensics": forensic_res
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
