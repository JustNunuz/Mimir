from core.rl_agent import RLAgent

_rl_agent = None

def get_rl_agent():
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent()
    return _rl_agent


def _score_to_label(score):
    """Maps a 0.0–1.0 score to a human-readable label with symmetric thresholds."""
    if score > 0.85:
        return "Likely AI Generated"
    elif score > 0.6:
        return "Possible AI Generated"
    elif score < 0.15:
        return "Likely Human Created"
    elif score < 0.4:
        return "Possible Human Created"
    else:
        return "Inconclusive"


def calculate_final_assessment(metadata_results, watermark_results, ai_results, forensic_results):
    """
    Calculates a final assessment score from multiple layers of evidence.

    The RL agent scores ALL cases (including hard-evidence cases) so its
    metadata/watermark weights actually get trained. Hard evidence then
    acts as a floor, boosting the score but never bypassing the agent.

    Returns (score, label, confidence).
    """
    # 1. Always let the RL agent score — it needs to see every case
    agent = get_rl_agent()
    features = agent.extract_features(metadata_results, watermark_results, ai_results, forensic_results)
    score = agent.predict(features)

    # 2. Hard evidence acts as a FLOOR, not a bypass
    hard_evidence_label = None
    if metadata_results and metadata_results.get("evidence_score", 0) > 0.8:
        score = max(score, 0.95)
        hard_evidence_label = "Verified Provenance Match (AI)"

    if watermark_results and watermark_results.get("watermark_found"):
        score = max(score, 0.95)
        hard_evidence_label = "Verified Watermark Match (AI)"

    # 3. Confidence: how far the score is from 0.5 (the neutral/uncertain midpoint)
    confidence = abs(score - 0.5) * 2.0

    # 4. Determine label — hard evidence overrides the label but not the scoring path
    if hard_evidence_label:
        label = hard_evidence_label
    else:
        label = _score_to_label(score)

    return float(score), label, float(confidence)
