import json
import copy
import sqlite3
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Safeguard constants
MIN_FEEDBACK_SAMPLES = 10      # Don't apply weight updates until this many samples
WEIGHT_BOUNDS = (-5.0, 5.0)    # Clamp weights to prevent runaway drift
MAX_HISTORY_SNAPSHOTS = 20     # Keep last N weight snapshots for rollback
INITIAL_LEARNING_RATE = 0.05
LR_DECAY = 0.998               # Gentle decay per batch application


class RLAgent:
    """
    A lightweight online-learning agent (logistic regression with SGD) that
    learns to weight detection-layer signals based on user feedback.

    Safety features:
    - Feedback is buffered; weights are only updated in batches
    - Weight clamping prevents runaway drift
    - Full weight history enables rollback after poisoning
    - Learning rate decays over time
    - Feedback is logged with timestamps for auditability
    """

    def __init__(self, db_path="mimir_registry.db"):
        self.db_path = db_path
        self.features = ['metadata_evidence', 'watermark_evidence', 'ai_prob', 'forensic_anomaly']
        self._init_db()
        self.weights, self.bias, self.learning_rate = self._load_state()
        self.feedback_buffer = []

    # ------------------------------------------------------------------ DB
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main weights table (single active row)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rl_weights (
                    id INTEGER PRIMARY KEY,
                    weights_json TEXT,
                    bias REAL,
                    learning_rate REAL DEFAULT 0.05
                )
            ''')

            # Weight history for rollback
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rl_weight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weights_json TEXT,
                    bias REAL,
                    learning_rate REAL,
                    timestamp TEXT,
                    reason TEXT
                )
            ''')

            # Feedback log for auditability
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rl_feedback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features_json TEXT,
                    target REAL,
                    prediction REAL,
                    timestamp TEXT
                )
            ''')

            # Insert default weights if empty
            cursor.execute("SELECT COUNT(*) FROM rl_weights")
            if cursor.fetchone()[0] == 0:
                default_weights = {
                    'metadata_evidence': 1.5,
                    'watermark_evidence': 2.0,
                    'ai_prob': 1.0,
                    'forensic_anomaly': 0.1
                }
                cursor.execute(
                    "INSERT INTO rl_weights (id, weights_json, bias, learning_rate) VALUES (1, ?, ?, ?)",
                    (json.dumps(default_weights), -0.5, INITIAL_LEARNING_RATE)
                )
            else:
                # Migrate: add learning_rate column if it doesn't exist
                try:
                    cursor.execute("SELECT learning_rate FROM rl_weights LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute("ALTER TABLE rl_weights ADD COLUMN learning_rate REAL DEFAULT 0.05")

            conn.commit()

    def _load_state(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT weights_json, bias, learning_rate FROM rl_weights WHERE id=1")
            row = cursor.fetchone()
            weights = json.loads(row[0])
            
            # Migrate old db keys to new names
            if 'ela_anomaly' in weights:
                weights['forensic_anomaly'] = weights.pop('ela_anomaly')
                
            bias = row[1]
            lr = row[2] if row[2] is not None else INITIAL_LEARNING_RATE
            return weights, bias, lr

    def load_weights(self):
        """Public API — returns (weights, bias) for backward compatibility."""
        w, b, _ = self._load_state()
        return w, b

    def _save_state(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE rl_weights SET weights_json=?, bias=?, learning_rate=? WHERE id=1",
                (json.dumps(self.weights), self.bias, self.learning_rate)
            )
            conn.commit()

    def save_weights(self):
        """Public API — backward compatible."""
        self._save_state()

    def _save_snapshot(self, reason="batch_update"):
        """Save a snapshot of current weights for rollback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO rl_weight_history (weights_json, bias, learning_rate, timestamp, reason) VALUES (?, ?, ?, ?, ?)",
                (json.dumps(self.weights), self.bias, self.learning_rate,
                 datetime.now().isoformat(), reason)
            )
            # Prune old snapshots beyond MAX_HISTORY_SNAPSHOTS
            cursor.execute(
                "DELETE FROM rl_weight_history WHERE id NOT IN "
                "(SELECT id FROM rl_weight_history ORDER BY id DESC LIMIT ?)",
                (MAX_HISTORY_SNAPSHOTS,)
            )
            conn.commit()

    def _log_feedback(self, features, target, prediction):
        """Log each feedback event for auditability."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO rl_feedback_log (features_json, target, prediction, timestamp) VALUES (?, ?, ?, ?)",
                    (json.dumps(features), target, prediction, datetime.now().isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to log feedback: %s", e)

    # --------------------------------------------------------- Features
    def extract_features(self, metadata_results, watermark_results, ai_results, forensic_results):
        meta = 1.0 if (metadata_results and metadata_results.get("evidence_score", 0) > 0.8) else 0.0
        water = 0.0
        if watermark_results:
            water = watermark_results.get("soft_signal", 1.0 if watermark_results.get("watermark_found") else 0.0)
        ai = ai_results.get("ai_probability", 0.5) if (ai_results and "error" not in ai_results) else 0.5
        ela = forensic_results.get("anomaly_score", 0.0) if (forensic_results and "error" not in forensic_results) else 0.0

        return {
            'metadata_evidence': meta,
            'watermark_evidence': water,
            'ai_prob': ai,
            'forensic_anomaly': ela
        }

    # --------------------------------------------------------- Predict
    def predict(self, feature_dict):
        """Calculate logit: w1*x1 + w2*x2 ... + b → sigmoid → probability."""
        logit = self.bias
        for f in self.features:
            logit += self.weights[f] * feature_dict[f]

        # Sigmoid to get probability
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        return float(prob)

    # --------------------------------------------------------- Learning
    def update_reward(self, feature_dict, is_ai_target):
        """
        Buffers feedback. When the buffer reaches MIN_FEEDBACK_SAMPLES,
        applies a batch gradient update with safety clamping.

        is_ai_target: 1.0 if the image was AI, 0.0 if human.
        """
        prediction = self.predict(feature_dict)
        self._log_feedback(feature_dict, is_ai_target, prediction)

        self.feedback_buffer.append((copy.deepcopy(feature_dict), float(is_ai_target)))

        if len(self.feedback_buffer) >= MIN_FEEDBACK_SAMPLES:
            self._apply_batch_update()
            return True  # Weights were updated
        else:
            remaining = MIN_FEEDBACK_SAMPLES - len(self.feedback_buffer)
            logger.info("Feedback buffered. %d more needed before weight update.", remaining)
            return False  # Buffered, not yet applied

    def _apply_batch_update(self):
        """Apply buffered feedback as a single batch gradient descent step."""
        # Snapshot before update for rollback
        self._save_snapshot(reason="pre_batch_update")

        for feature_dict, target in self.feedback_buffer:
            prob = self.predict(feature_dict)
            error = target - prob

            # Update bias
            self.bias += self.learning_rate * error

            # Update weights
            for f in self.features:
                self.weights[f] += self.learning_rate * error * feature_dict[f]

        # Clamp weights to prevent runaway drift
        for f in self.features:
            self.weights[f] = float(np.clip(self.weights[f], *WEIGHT_BOUNDS))
        self.bias = float(np.clip(self.bias, *WEIGHT_BOUNDS))

        # Decay learning rate
        self.learning_rate *= LR_DECAY

        self.feedback_buffer.clear()
        self._save_state()
        logger.info("Batch update applied. New weights: %s, bias: %.4f, lr: %.6f",
                     self.weights, self.bias, self.learning_rate)

    def get_buffer_status(self):
        """Returns info about the current feedback buffer for UI display."""
        return {
            "buffered": len(self.feedback_buffer),
            "needed": MIN_FEEDBACK_SAMPLES,
            "remaining": max(0, MIN_FEEDBACK_SAMPLES - len(self.feedback_buffer)),
        }

    # --------------------------------------------------------- Rollback
    def rollback(self, steps=1):
        """
        Roll back weights to a previous snapshot.
        Returns True if rollback succeeded.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT weights_json, bias, learning_rate FROM rl_weight_history "
                "ORDER BY id DESC LIMIT 1 OFFSET ?",
                (steps - 1,)
            )
            row = cursor.fetchone()
            if row is None:
                logger.warning("Rollback failed: no snapshot at offset %d", steps - 1)
                return False

            self.weights = json.loads(row[0])
            self.bias = row[1]
            self.learning_rate = row[2] if row[2] else INITIAL_LEARNING_RATE
            self._save_state()
            self._save_snapshot(reason=f"rollback_{steps}_steps")
            logger.info("Rolled back %d step(s). Weights: %s", steps, self.weights)
            return True

    def reset_to_defaults(self):
        """Hard reset weights to the initial defaults."""
        self._save_snapshot(reason="pre_reset")
        self.weights = {
            'metadata_evidence': 1.5,
            'watermark_evidence': 2.0,
            'ai_prob': 1.0,
            'forensic_anomaly': 0.1
        }
        self.bias = -0.5
        self.learning_rate = INITIAL_LEARNING_RATE
        self.feedback_buffer.clear()
        self._save_state()
        logger.info("Weights reset to defaults.")
