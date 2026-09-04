import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
    phash = Column(String)
    dhash = Column(String)
    ahash = Column(String)
    watermark_id = Column(String)
    metadata_json = Column(Text)
    ai_score = Column(Float)
    human_score = Column(Float)
    final_assessment = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class RegistryDB:
    def __init__(self, db_url=None):
        # Default to local sqlite if no PostgreSQL/external URL is provided
        # This addresses Reviewer 2's scalability concern by allowing PostgreSQL deployments
        if db_url is None:
            db_url = os.environ.get("DATABASE_URL", "sqlite:///mimir_registry.db")
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_record(self, filename, phash, dhash, ahash, watermark_id, metadata, ai_score, human_score, final_assessment):
        with self.Session() as session:
            record = ImageRecord(
                filename=filename,
                phash=phash,
                dhash=dhash,
                ahash=ahash,
                watermark_id=watermark_id,
                metadata_json=json.dumps(metadata) if metadata else "{}",
                ai_score=ai_score,
                human_score=human_score,
                final_assessment=final_assessment,
                timestamp=datetime.now()
            )
            session.add(record)
            session.commit()
            return record.id

    def find_by_hash(self, hash_type, hash_value, max_distance=5):
        import imagehash
        with self.Session() as session:
            # Safely getattr for the hash type column
            column = getattr(ImageRecord, hash_type, None)
            if column is None:
                return []
            
            # TODO: O(n) Python loop scan won't scale beyond a few thousand rows.
            # Move to a BK-tree or LSH-bucket index once the registry grows.
            records = session.query(ImageRecord).all()
            matches = []
            try:
                target_hash = imagehash.hex_to_hash(hash_value)
            except Exception:
                return []
                
            for r in records:
                val = getattr(r, hash_type, None)
                if val:
                    try:
                        h = imagehash.hex_to_hash(val)
                        if target_hash - h <= max_distance:
                            matches.append(self._record_to_dict(r))
                    except Exception:
                        pass
            return matches
    
    def get_all_records(self):
        with self.Session() as session:
            records = session.query(ImageRecord).order_by(ImageRecord.timestamp.desc()).all()
            return [self._record_to_dict(r) for r in records]

    def _record_to_dict(self, record):
        return {
            "id": record.id,
            "filename": record.filename,
            "phash": record.phash,
            "dhash": record.dhash,
            "ahash": record.ahash,
            "watermark_id": record.watermark_id,
            "metadata_json": record.metadata_json,
            "ai_score": record.ai_score,
            "human_score": record.human_score,
            "final_assessment": record.final_assessment,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None
        }
