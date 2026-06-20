import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class AIImageDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # We now use a real, pre-trained Vision Transformer model for AI detection
        try:
            model_name = "umm-maybe/AI-image-detector"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            self.error = str(e)

    def predict(self, image_file):
        if not self.is_loaded:
            return {"error": "Model not loaded: " + getattr(self, 'error', 'Unknown Error')}
            
        try:
            image_file.seek(0)
            img = Image.open(image_file).convert('RGB')
            
            inputs = self.processor(img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                
            # For this specific model: {0: 'artificial', 1: 'human'}
            ai_prob = float(probabilities[0])
            human_prob = float(probabilities[1])
            
            return {
                "ai_probability": ai_prob,
                "human_probability": human_prob,
                "confidence": abs(ai_prob - human_prob)
            }
            
        except Exception as e:
            return {"error": str(e)}

ai_detector = None
def get_ai_detector():
    global ai_detector
    if ai_detector is None:
        ai_detector = AIImageDetector()
    return ai_detector
