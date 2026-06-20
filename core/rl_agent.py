import json
import sqlite3
import numpy as np

class RLAgent:
    """
    A lightweight Reinforcement Learning agent (Contextual Bandit) that uses 
    incremental gradient descent to update layer weights based on user feedback.
    """
    def __init__(self, db_path="mimir_registry.db", learning_rate=0.05):
        self.db_path = db_path
        self.learning_rate = learning_rate
        self.features = ['metadata_evidence', 'watermark_evidence', 'ai_prob', 'ela_anomaly']
        self._init_db()
        self.weights, self.bias = self.load_weights()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rl_weights (
                    id INTEGER PRIMARY KEY,
                    weights_json TEXT,
                    bias REAL
                )
            ''')
            # Insert default weights if empty
            cursor.execute("SELECT COUNT(*) FROM rl_weights")
            if cursor.fetchone()[0] == 0:
                default_weights = {
                    'metadata_evidence': 1.5,
                    'watermark_evidence': 2.0,
                    'ai_prob': 1.0,
                    'ela_anomaly': 0.1
                }
                cursor.execute(
                    "INSERT INTO rl_weights (id, weights_json, bias) VALUES (1, ?, ?)",
                    (json.dumps(default_weights), -0.5)
                )
            conn.commit()

    def load_weights(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT weights_json, bias FROM rl_weights WHERE id=1")
            row = cursor.fetchone()
            weights = json.loads(row[0])
            bias = row[1]
            return weights, bias

    def save_weights(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE rl_weights SET weights_json=?, bias=? WHERE id=1",
                (json.dumps(self.weights), self.bias)
            )
            conn.commit()

    def extract_features(self, metadata_results, watermark_results, ai_results, forensic_results):
        meta = 1.0 if (metadata_results and metadata_results.get("evidence_score", 0) > 0.8) else 0.0
        water = 1.0 if (watermark_results and watermark_results.get("watermark_found")) else 0.0
        ai = ai_results.get("ai_probability", 0.5) if (ai_results and "error" not in ai_results) else 0.5
        ela = forensic_results.get("anomaly_score", 0.0) if (forensic_results and "error" not in forensic_results) else 0.0
        
        return {
            'metadata_evidence': meta,
            'watermark_evidence': water,
            'ai_prob': ai,
            'ela_anomaly': ela
        }

    def predict(self, feature_dict):
        # Calculate logit: w1*x1 + w2*x2 ... + b
        logit = self.bias
        for f in self.features:
            logit += self.weights[f] * feature_dict[f]
            
        # Sigmoid to get probability
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob

    def update_reward(self, feature_dict, is_ai_target):
        """
        Feedback loop: Updates the weights based on whether the image was actually AI or not.
        is_ai_target: 1.0 if the image was AI, 0.0 if human.
        Uses logistic regression gradient descent step.
        """
        prob = self.predict(feature_dict)
        error = is_ai_target - prob # Positive if we underestimated AI, negative if we overestimated
        
        # Update bias
        self.bias += self.learning_rate * error
        
        # Update weights
        for f in self.features:
            # Gradient is error * feature_value
            self.weights[f] += self.learning_rate * error * feature_dict[f]
            
        self.save_weights()
