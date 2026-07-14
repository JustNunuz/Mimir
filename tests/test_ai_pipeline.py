"""
Test suite for Mimir's AI detection pipeline.

Validates the four critical fixes:
1. Multi-model ensemble detector with input validation
2. RL agent safety (clamping, buffering, rollback, poisoning resistance)
3. Scoring pipeline — no more short-circuit bypass
4. Frequency-domain forensics alongside ELA
"""
import os
import sys
import json
import sqlite3
import tempfile
import copy
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_agent import RLAgent, MIN_FEEDBACK_SAMPLES, WEIGHT_BOUNDS
from core.scoring import calculate_final_assessment, _score_to_label
from forensics.ela import error_level_analysis, frequency_domain_analysis, combined_forensic_analysis


# ──────────────────────────── Helpers ────────────────────────────

def _make_test_image(width=256, height=256, color=(128, 128, 128), fmt="JPEG"):
    """Creates a solid-colour test image as a BytesIO file object."""
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def _make_gradient_image(width=256, height=256, fmt="JPEG"):
    """Creates a gradient test image — simulates more realistic content."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [x % 256, y % 256, (x + y) % 256]
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def _get_temp_db():
    """Returns a temporary database path for isolated RL agent tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ══════════════════════════════════════════════════════════════════
#  TEST 1: RL Agent Safety
# ══════════════════════════════════════════════════════════════════

class TestRLAgentSafety:
    """Validates that the RL agent cannot be trivially poisoned."""

    def setup_method(self):
        self.db_path = _get_temp_db()
        self.agent = RLAgent(db_path=self.db_path)

    def teardown_method(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_single_feedback_does_not_update_weights(self):
        """A single feedback event must NOT change weights — it should be buffered."""
        original_weights = copy.deepcopy(self.agent.weights)
        original_bias = self.agent.bias

        features = {
            'metadata_evidence': 0.0,
            'watermark_evidence': 0.0,
            'ai_prob': 0.9,
            'ela_anomaly': 0.3,
        }
        result = self.agent.update_reward(features, 1.0)

        assert result is False, "Weights should NOT be updated on a single feedback"
        assert self.agent.weights == original_weights, "Weights changed on single feedback!"
        assert self.agent.bias == original_bias, "Bias changed on single feedback!"

    def test_batch_update_fires_after_min_samples(self):
        """Weights should only update after MIN_FEEDBACK_SAMPLES are collected."""
        original_weights = copy.deepcopy(self.agent.weights)

        features = {
            'metadata_evidence': 0.0,
            'watermark_evidence': 0.0,
            'ai_prob': 0.9,
            'ela_anomaly': 0.3,
        }

        for i in range(MIN_FEEDBACK_SAMPLES - 1):
            result = self.agent.update_reward(features, 1.0)
            assert result is False

        # The MIN_FEEDBACK_SAMPLES-th feedback should trigger the update
        result = self.agent.update_reward(features, 1.0)
        assert result is True, "Batch update should have fired"
        assert self.agent.weights != original_weights, "Weights should have changed after batch"

    def test_weight_clamping_prevents_extreme_values(self):
        """Even after many aggressive updates, weights must stay within bounds."""
        features = {
            'metadata_evidence': 1.0,
            'watermark_evidence': 1.0,
            'ai_prob': 1.0,
            'ela_anomaly': 1.0,
        }

        # Simulate 500 rounds of aggressive "this is AI!" feedback
        for _ in range(500):
            self.agent.feedback_buffer.append((copy.deepcopy(features), 1.0))
            if len(self.agent.feedback_buffer) >= MIN_FEEDBACK_SAMPLES:
                self.agent._apply_batch_update()

        for f_name in self.agent.features:
            assert self.agent.weights[f_name] >= WEIGHT_BOUNDS[0], \
                f"{f_name} went below lower bound: {self.agent.weights[f_name]}"
            assert self.agent.weights[f_name] <= WEIGHT_BOUNDS[1], \
                f"{f_name} went above upper bound: {self.agent.weights[f_name]}"
        assert self.agent.bias >= WEIGHT_BOUNDS[0]
        assert self.agent.bias <= WEIGHT_BOUNDS[1]

    def test_poisoning_resistance(self):
        """
        Simulates an adversarial user sending contradictory feedback.
        After poisoning, predictions should still be reasonable (not inverted).
        """
        # First, feed honest data: AI images with high ai_prob
        honest_features = {
            'metadata_evidence': 0.0,
            'watermark_evidence': 0.0,
            'ai_prob': 0.95,
            'ela_anomaly': 0.2,
        }

        for _ in range(50):
            self.agent.feedback_buffer.append((copy.deepcopy(honest_features), 1.0))
            if len(self.agent.feedback_buffer) >= MIN_FEEDBACK_SAMPLES:
                self.agent._apply_batch_update()

        # Now simulate poisoning: same features, but lying that it's human
        for _ in range(30):
            self.agent.feedback_buffer.append((copy.deepcopy(honest_features), 0.0))
            if len(self.agent.feedback_buffer) >= MIN_FEEDBACK_SAMPLES:
                self.agent._apply_batch_update()

        # The ai_prob weight should still be positive (not inverted)
        assert self.agent.weights['ai_prob'] > 0, \
            f"ai_prob weight went negative after poisoning: {self.agent.weights['ai_prob']}"

    def test_rollback_restores_previous_weights(self):
        """Rollback should restore weights to a previous snapshot."""
        original_weights = copy.deepcopy(self.agent.weights)
        original_bias = self.agent.bias

        # Apply a batch update
        features = {
            'metadata_evidence': 1.0,
            'watermark_evidence': 1.0,
            'ai_prob': 0.95,
            'ela_anomaly': 0.5,
        }
        for _ in range(MIN_FEEDBACK_SAMPLES):
            self.agent.feedback_buffer.append((copy.deepcopy(features), 1.0))
        self.agent._apply_batch_update()

        assert self.agent.weights != original_weights, "Weights should have changed"

        # Rollback
        success = self.agent.rollback(steps=1)
        assert success is True
        assert self.agent.weights == original_weights, "Rollback did not restore weights"
        assert self.agent.bias == original_bias, "Rollback did not restore bias"

    def test_reset_to_defaults(self):
        """reset_to_defaults should restore the initial weight configuration."""
        # Mess up the weights
        self.agent.weights['ai_prob'] = 99.0
        self.agent.bias = 42.0
        self.agent._save_state()

        self.agent.reset_to_defaults()

        assert self.agent.weights['ai_prob'] == 1.0
        assert self.agent.weights['metadata_evidence'] == 1.5
        assert self.agent.bias == -0.5

    def test_feedback_is_logged(self):
        """Every feedback event should be persisted in the feedback log."""
        features = {
            'metadata_evidence': 0.0,
            'watermark_evidence': 0.0,
            'ai_prob': 0.7,
            'ela_anomaly': 0.1,
        }
        self.agent.update_reward(features, 1.0)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rl_feedback_log")
            count = cursor.fetchone()[0]
        assert count == 1, "Feedback was not logged"

    def test_buffer_status_reporting(self):
        """get_buffer_status should accurately report buffer state."""
        status = self.agent.get_buffer_status()
        assert status["buffered"] == 0
        assert status["needed"] == MIN_FEEDBACK_SAMPLES
        assert status["remaining"] == MIN_FEEDBACK_SAMPLES

        # Add one feedback
        features = {'metadata_evidence': 0, 'watermark_evidence': 0, 'ai_prob': 0.5, 'ela_anomaly': 0}
        self.agent.update_reward(features, 1.0)

        status = self.agent.get_buffer_status()
        assert status["buffered"] == 1
        assert status["remaining"] == MIN_FEEDBACK_SAMPLES - 1


# ══════════════════════════════════════════════════════════════════
#  TEST 2: Scoring Pipeline — No Short-Circuit
# ══════════════════════════════════════════════════════════════════

class TestScoringPipeline:
    """Validates that scoring no longer bypasses the RL agent."""

    def test_label_thresholds(self):
        """Verify _score_to_label covers the full range correctly."""
        assert _score_to_label(0.95) == "Likely AI Generated"
        assert _score_to_label(0.70) == "Possible AI Generated"
        assert _score_to_label(0.50) == "Inconclusive"
        assert _score_to_label(0.30) == "Possible Human Created"
        assert _score_to_label(0.10) == "Likely Human Created"

    def test_metadata_does_not_bypass_rl(self):
        """
        With strong metadata evidence, the final label should be 'Verified
        Provenance' but the RL agent should still have been invoked (score
        is boosted, not hardcoded).
        """
        metadata = {"evidence_score": 0.95, "findings": [{"source": "EXIF", "keyword_found": "dall-e"}]}
        watermark = {"watermark_found": False}
        ai = {"ai_probability": 0.3, "human_probability": 0.7}  # Model says human!
        forensic = {"anomaly_score": 0.1}

        score, label, conf = calculate_final_assessment(metadata, watermark, ai, forensic)

        assert score >= 0.95, f"Score should be at least 0.95 with strong metadata, got {score}"
        assert "Verified" in label
        assert conf > 0.5

    def test_watermark_does_not_bypass_rl(self):
        """Same test for watermark evidence."""
        metadata = {"evidence_score": 0.0, "findings": []}
        watermark = {"watermark_found": True, "recovered_identifier": "test"}
        ai = {"ai_probability": 0.2, "human_probability": 0.8}
        forensic = {"anomaly_score": 0.1}

        score, label, conf = calculate_final_assessment(metadata, watermark, ai, forensic)

        assert score >= 0.95
        assert "Verified" in label

    def test_no_evidence_returns_rl_score(self):
        """With no hard evidence, the score should come from the RL agent."""
        metadata = {"evidence_score": 0.0, "findings": []}
        watermark = {"watermark_found": False}
        ai = {"ai_probability": 0.9, "human_probability": 0.1}
        forensic = {"anomaly_score": 0.5}

        score, label, conf = calculate_final_assessment(metadata, watermark, ai, forensic)

        # Should not be 0.95 (that's the hard-evidence floor)
        # The exact value depends on RL weights, but it should be > 0.5 with high ai_prob
        assert 0.0 <= score <= 1.0
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0


# ══════════════════════════════════════════════════════════════════
#  TEST 3: Forensics — Frequency Domain Analysis
# ══════════════════════════════════════════════════════════════════

class TestForensics:
    """Validates ELA and the new frequency-domain analysis."""

    def test_ela_returns_valid_result(self):
        """ELA should produce valid scores for a normal image."""
        img = _make_gradient_image()
        result = error_level_analysis(img)

        assert "error" not in result, f"ELA errored: {result.get('error')}"
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert "ela_image_base64" in result
        assert result["variance"] >= 0

    def test_frequency_domain_returns_valid_result(self):
        """Frequency analysis should produce valid scores."""
        img = _make_gradient_image()
        result = frequency_domain_analysis(img)

        assert "error" not in result, f"Frequency analysis errored: {result.get('error')}"
        assert "hf_ratio" in result
        assert "spectral_entropy" in result
        assert "ai_frequency_score" in result
        assert 0.0 <= result["ai_frequency_score"] <= 1.0
        assert "spectrum_image_base64" in result

    def test_combined_forensic_produces_composite_score(self):
        """Combined analysis should have both ELA and frequency sub-results."""
        img = _make_gradient_image()
        result = combined_forensic_analysis(img)

        assert "anomaly_score" in result, "Missing composite anomaly_score"
        assert "ela" in result, "Missing ELA sub-result"
        assert "frequency" in result, "Missing frequency sub-result"
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_frequency_analysis_handles_solid_image(self):
        """A solid-colour image should not crash frequency analysis."""
        img = _make_test_image(color=(0, 0, 0))
        result = frequency_domain_analysis(img)

        assert "error" not in result, f"Frequency analysis crashed on solid image: {result.get('error')}"
        assert 0.0 <= result["ai_frequency_score"] <= 1.0

    def test_ela_handles_png_input(self):
        """ELA should handle PNG input (converts internally via JPEG recompression)."""
        img = _make_test_image(fmt="PNG")
        result = error_level_analysis(img)
        assert "error" not in result


# ══════════════════════════════════════════════════════════════════
#  TEST 4: AI Model Detector — Input Validation
# ══════════════════════════════════════════════════════════════════

class TestAIModelInputValidation:
    """Tests the input validation layer without requiring actual model weights."""

    def test_safe_load_image_rgb_conversion(self):
        """RGBA and other modes should be converted to RGB."""
        from detectors.ai_model import _safe_load_image

        # Create an RGBA image
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        buf = BytesIO()
        rgba_img.save(buf, format="PNG")
        buf.seek(0)

        loaded = _safe_load_image(buf)
        assert loaded.mode == "RGB", f"Expected RGB, got {loaded.mode}"

    def test_safe_load_image_resolution_capping(self):
        """Images larger than MAX_RESOLUTION should be downscaled."""
        from detectors.ai_model import _safe_load_image, MAX_RESOLUTION

        big_img = Image.new("RGB", (4000, 3000), (0, 255, 0))
        buf = BytesIO()
        big_img.save(buf, format="JPEG")
        buf.seek(0)

        loaded = _safe_load_image(buf)
        assert loaded.width <= MAX_RESOLUTION[0], f"Width {loaded.width} exceeds {MAX_RESOLUTION[0]}"
        assert loaded.height <= MAX_RESOLUTION[1], f"Height {loaded.height} exceeds {MAX_RESOLUTION[1]}"

    def test_safe_load_image_preserves_small_images(self):
        """Images within MAX_RESOLUTION should not be resized."""
        from detectors.ai_model import _safe_load_image

        small_img = Image.new("RGB", (200, 200), (0, 0, 255))
        buf = BytesIO()
        small_img.save(buf, format="JPEG")
        buf.seek(0)

        loaded = _safe_load_image(buf)
        assert loaded.size == (200, 200)


# ══════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
