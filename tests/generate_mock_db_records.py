import os
import random
import uuid
from datetime import datetime, timedelta
import json

# Ensure we can import from the parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry.db import RegistryDB

def generate_mock_records(num_records=1000):
    print(f"Generating {num_records} mock records for the Registry Database...")
    db = RegistryDB()
    
    # Probabilities for different categories
    categories = [
        {"label": "Likely Human Created", "score_range": (0.01, 0.14), "weight": 0.40},
        {"label": "Possible Human Created", "score_range": (0.15, 0.39), "weight": 0.20},
        {"label": "Inconclusive", "score_range": (0.40, 0.60), "weight": 0.10},
        {"label": "Possible AI Generated", "score_range": (0.61, 0.85), "weight": 0.10},
        {"label": "Likely AI Generated", "score_range": (0.86, 0.94), "weight": 0.10},
        {"label": "Verified Provenance Match (AI)", "score_range": (0.95, 0.99), "weight": 0.05},
        {"label": "Verified Watermark Match (AI)", "score_range": (0.95, 0.99), "weight": 0.05},
    ]
    
    labels = [c["label"] for c in categories]
    weights = [c["weight"] for c in categories]
    
    for i in range(num_records):
        # Pick a category based on weights
        chosen = random.choices(categories, weights=weights, k=1)[0]
        
        ai_score = random.uniform(chosen["score_range"][0], chosen["score_range"][1])
        human_score = 1.0 - ai_score
        
        # Generate fake hashes
        phash = uuid.uuid4().hex[:16]
        dhash = uuid.uuid4().hex[:16]
        ahash = uuid.uuid4().hex[:16]
        
        # Fake metadata
        metadata = {}
        if "Provenance" in chosen["label"]:
            metadata = {"C2PA": {"generator": "Midjourney v6", "software": "AI"}}
            
        watermark_id = None
        if "Watermark" in chosen["label"]:
            watermark_id = f"WM-{uuid.uuid4().hex[:8].upper()}"
            
        # Random timestamp over the past 30 days
        days_ago = random.uniform(0, 30)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        filename = f"sample_image_{i:04d}.jpg"
        
        # We need to insert it manually to allow custom timestamps,
        # or we can just use the db.add_record and let it use current time.
        # But spreading out timestamps is better for realistic UI tests.
        with db.Session() as session:
            from registry.db import ImageRecord
            record = ImageRecord(
                filename=filename,
                phash=phash,
                dhash=dhash,
                ahash=ahash,
                watermark_id=watermark_id,
                metadata_json=json.dumps(metadata),
                ai_score=ai_score,
                human_score=human_score,
                final_assessment=chosen["label"],
                timestamp=timestamp
            )
            session.add(record)
            session.commit()
            
        if (i + 1) % 100 == 0:
            print(f"Inserted {i + 1} records...")
            
    print(f"Successfully populated database with {num_records} records.")

if __name__ == '__main__':
    generate_mock_records(1000)
