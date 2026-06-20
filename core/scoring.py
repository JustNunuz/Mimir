from core.rl_agent import RLAgent

_rl_agent = None

def get_rl_agent():
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent()
    return _rl_agent


def calculate_final_assessment(metadata_results, watermark_results, ai_results, forensic_results):
    """
    Calculates a final assessment score from multiple layers of evidence.
    The RL agent dynamically weights the layer signals based on past feedback.
    Returns a score from 0.0 (Verified Human) to 1.0 (Verified AI), and a categorical label.
    """
    # 1. Hard Evidence (Metadata & Watermarks) — always highest priority
    if metadata_results and metadata_results.get("evidence_score", 0) > 0.8:
        return 0.95, "Verified Provenance Match (AI)", 0.95
        
    if watermark_results and watermark_results.get("watermark_found"):
        return 0.95, "Verified Watermark Match (AI)", 0.95

    # 2. RL Agent: probabilistic scoring using learned weights
    agent = get_rl_agent()
    features = agent.extract_features(metadata_results, watermark_results, ai_results, forensic_results)
    score = agent.predict(features)

    # Confidence: how far the score is from 0.5 (the neutral/uncertain midpoint)
    confidence = abs(score - 0.5) * 2.0

    # Determine Label
    if score > 0.85:
        label = "Likely AI Generated"
    elif score > 0.6:
        label = "Possible AI Generated"
    elif score < 0.15:
        label = "Likely Human Created"
    elif score < 0.4:
        label = "Possible Human Created"
    else:
        label = "Inconclusive"

    return float(score), label, float(confidence)
