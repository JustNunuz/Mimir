import torch
import numpy as np
import concurrent.futures
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Maximum resolution to prevent OOM — images larger than this are downscaled
MAX_RESOLUTION = (1024, 1024)

# Model registry: architecturally diverse models for ensemble voting
MODEL_REGISTRY = [
    {
        "name": "umm-maybe/AI-image-detector",
        "label_map": {0: "artificial", 1: "human"},
        "ai_index": 0,
    },
    {
        "name": "Organika/sdxl-detector",
        "label_map": {0: "artificial", 1: "human"},
        "ai_index": 0,
    },
]


def _safe_load_image(image_file):
    """
    Validates and normalises an image for inference:
    - Converts to RGB (handles CMYK, RGBA, palette, 16-bit, etc.)
    - Caps resolution to MAX_RESOLUTION to prevent OOM
    """
    image_file.seek(0)
    img = Image.open(image_file)

    # Force RGB regardless of source colour space
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Down-scale oversized images while preserving aspect ratio
    if img.width > MAX_RESOLUTION[0] or img.height > MAX_RESOLUTION[1]:
        img.thumbnail(MAX_RESOLUTION, Image.LANCZOS)

    return img


class _SingleModelDetector:
    """Wraps a single HuggingFace image-classification checkpoint."""

    def __init__(self, model_info, device):
        self.model_name = model_info["name"]
        self.ai_index = model_info["ai_index"]
        self.is_loaded = False
        self.error = None

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(device)
            self.device = device
            self.is_loaded = True
            logger.info("Loaded model: %s", self.model_name)
        except Exception as e:
            self.error = str(e)
            logger.warning("Failed to load model %s: %s", self.model_name, e)

    def predict(self, img: Image.Image):
        """Returns (ai_probability, human_probability) or None on failure."""
        if not self.is_loaded:
            return None

        try:
            inputs = self.processor(img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

            ai_prob = float(probs[self.ai_index])
            human_prob = float(1.0 - ai_prob)
            return ai_prob, human_prob
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", self.model_name, e)
            return None


class AIImageDetector:
    """
    Multi-model ensemble detector.

    Loads every model in MODEL_REGISTRY and aggregates their predictions.
    Confidence is derived from inter-model agreement, not just the
    probability value — disagreement among models signals genuine
    uncertainty.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load all models concurrently to cut startup time
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_REGISTRY)) as pool:
            futures = [pool.submit(_SingleModelDetector, info, self.device) for info in MODEL_REGISTRY]
            self.models = [f.result() for f in futures]

        self.loaded_count = sum(1 for m in self.models if m.is_loaded)
        if self.loaded_count == 0:
            self.is_loaded = False
            self.error = "No models loaded successfully"
        else:
            self.is_loaded = True
            self.error = None

    def predict(self, image_file):
        if not self.is_loaded:
            return {"error": "Model not loaded: " + (self.error or "Unknown Error")}

        try:
            img = _safe_load_image(image_file)
        except Exception as e:
            return {"error": f"Image preprocessing failed: {e}"}

        # Collect predictions from every loaded model
        predictions = []
        model_details = []
        for m in self.models:
            result = m.predict(img)
            if result is not None:
                ai_prob, human_prob = result
                predictions.append(ai_prob)
                model_details.append({
                    "model": m.model_name,
                    "ai_probability": ai_prob,
                    "human_probability": human_prob,
                })

        if not predictions:
            return {"error": "All models failed during prediction"}

        # Ensemble: mean probability across models
        mean_ai = float(np.mean(predictions))
        mean_human = float(1.0 - mean_ai)

        # Confidence from inter-model agreement:
        # Low std-dev = high agreement = high confidence
        # With a single model, fall back to distance-from-0.5
        if len(predictions) >= 2:
            agreement = 1.0 - float(np.std(predictions)) * 2.0  # std ∈ [0, 0.5] → agreement ∈ [0, 1]
            distance_conf = abs(mean_ai - 0.5) * 2.0
            confidence = float(np.clip(min(agreement, distance_conf), 0.0, 1.0))
        else:
            confidence = abs(mean_ai - 0.5) * 2.0

        return {
            "ai_probability": mean_ai,
            "human_probability": mean_human,
            "confidence": confidence,
            "models_used": len(predictions),
            "model_details": model_details,
        }


ai_detector = None
def get_ai_detector():
    global ai_detector
    if ai_detector is None:
        ai_detector = AIImageDetector()
    return ai_detector
