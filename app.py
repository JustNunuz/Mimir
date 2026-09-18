import streamlit as st
import pandas as pd
import io
import concurrent.futures
import threading
import base64

from detectors.metadata import analyze_metadata
from detectors.watermark import detect_watermark
from detectors.hashing import generate_hashes
from detectors.ai_model import get_ai_detector
from forensics.ela import combined_forensic_analysis
from core.scoring import calculate_final_assessment, get_rl_agent
from registry.db import RegistryDB

st.set_page_config(page_title="Mimir AI Detection", page_icon="👁️", layout="wide")

db = RegistryDB()

# ---- Background model loading ------------------------------------------------
# Kick off model loading in a daemon thread so the page renders immediately.

@st.cache_resource(show_spinner=False)
def _start_detector_load():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mimir_model")
    future = executor.submit(get_ai_detector)
    return future, executor

_detector_future, _detector_executor = _start_detector_load()

def _get_detector():
    """Returns detector; blocks only if still loading."""
    return _detector_future.result()

def _detector_ready():
    return _detector_future.done()

# ---- Ground-truth RL targets -------------------------------------------------
_LABEL_TO_TARGET = {
    "AI":    0.95,
    "Human": 0.05,
}

# ==============================================================================
st.title("👁️ Mimir: AI Image Provenance & Detection")
st.markdown("Analyze images using a multi-layered approach to detect AI generation and provenance.")

tab1, tab2, tab3 = st.tabs(["🔍 Scan Image", "📚 Registry Explorer", "📊 System Statistics"])

# ==============================================================================
with tab1:

    # Model readiness badge
    if _detector_ready():
        st.success("🟢 AI models loaded and ready")
    else:
        st.info("⏳ AI models are loading in the background… You can still scan; it will wait automatically.")

    uploaded_file = st.file_uploader("Upload an image to analyze...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Run Full Scan", key="btn_scan"):
            file_bytes = uploaded_file.getvalue()

            # --- Step 1: fast hash computation + registry lookup ---
            with st.spinner("Checking image registry…"):
                hash_res = generate_hashes(io.BytesIO(file_bytes))

            phash = hash_res.get("phash")
            known_match = None
            if phash:
                matches = db.find_by_hash("phash", phash, max_distance=8)
                labeled = [m for m in matches if m.get("user_label")]
                if labeled:
                    # Most recent labeled match wins
                    known_match = sorted(labeled, key=lambda m: m.get("timestamp", ""), reverse=True)[0]

            if known_match:
                # === FAST PATH: image was previously ground-truth labelled ===
                known_label = known_match["user_label"]  # 'AI' or 'Human'
                final_score = 0.95 if known_label == "AI" else 0.05
                label = "Likely AI Generated" if known_label == "AI" else "Likely Human Created"
                conf = 0.99

                record_id = db.add_record(
                    filename=uploaded_file.name,
                    phash=hash_res.get("phash"),
                    dhash=hash_res.get("dhash"),
                    ahash=hash_res.get("ahash"),
                    watermark_id=known_match.get("watermark_id"),
                    metadata=None,
                    ai_score=final_score,
                    human_score=1.0 - final_score,
                    final_assessment=label,
                    user_label=known_label,
                )

                st.session_state.update({
                    "scan_complete": True,
                    "scan_mode": "cached",
                    "last_label": label,
                    "last_score": final_score,
                    "last_conf": conf,
                    "last_record_id": record_id,
                    "last_features": None,
                    "last_hash_res": hash_res,
                    "last_ai_res": {},
                    "last_forensic_res": {},
                    "last_metadata_res": {},
                    "last_watermark_res": {},
                    "known_label": known_label,
                })

            else:
                # === FULL SCAN PATH ===
                with st.spinner("Analyzing all layers in parallel…"):
                    detector = _get_detector()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        f_meta     = executor.submit(analyze_metadata,        io.BytesIO(file_bytes))
                        f_water    = executor.submit(detect_watermark,         io.BytesIO(file_bytes))
                        f_ai       = executor.submit(detector.predict,         io.BytesIO(file_bytes))
                        f_forensic = executor.submit(combined_forensic_analysis, io.BytesIO(file_bytes))

                        metadata_res  = f_meta.result()
                        watermark_res = f_water.result()
                        ai_res        = f_ai.result()
                        forensic_res  = f_forensic.result()

                final_score, label, conf = calculate_final_assessment(
                    metadata_res, watermark_res, ai_res, forensic_res
                )

                record_id = db.add_record(
                    filename=uploaded_file.name,
                    phash=hash_res.get("phash"),
                    dhash=hash_res.get("dhash"),
                    ahash=hash_res.get("ahash"),
                    watermark_id=watermark_res.get("recovered_identifier"),
                    metadata=metadata_res.get("findings"),
                    ai_score=final_score,
                    human_score=1.0 - final_score,
                    final_assessment=label,
                )

                agent = get_rl_agent()
                st.session_state.update({
                    "scan_complete": True,
                    "scan_mode": "full",
                    "last_label": label,
                    "last_score": final_score,
                    "last_conf": conf,
                    "last_record_id": record_id,
                    "last_features": agent.extract_features(metadata_res, watermark_res, ai_res, forensic_res),
                    "last_hash_res": hash_res,
                    "last_ai_res": ai_res,
                    "last_forensic_res": forensic_res,
                    "last_metadata_res": metadata_res,
                    "last_watermark_res": watermark_res,
                    "known_label": None,
                })

    # ---- Results (shown via session state so they persist across rerenders) ----
    if st.session_state.get("scan_complete"):
        label        = st.session_state["last_label"]
        final_score  = st.session_state["last_score"]
        conf         = st.session_state["last_conf"]
        record_id    = st.session_state["last_record_id"]
        scan_mode    = st.session_state["scan_mode"]
        hash_res     = st.session_state["last_hash_res"]
        ai_res       = st.session_state["last_ai_res"]
        forensic_res = st.session_state["last_forensic_res"]
        metadata_res = st.session_state["last_metadata_res"]
        watermark_res= st.session_state["last_watermark_res"]

        if scan_mode == "cached":
            st.info(f"⚡ **Known Image** — previously verified as **{st.session_state['known_label']}**. "
                    "Result returned from registry instantly.")
        else:
            st.success("Scan Complete!")

        st.header("Results Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assessment", label)
        with col2:
            st.metric("AI Probability", f"{final_score:.1%}")
        with col3:
            st.metric("Confidence", f"{conf:.1%}")

        if scan_mode == "full":
            # Show ensemble details
            if ai_res.get("model_details"):
                with st.expander("🔬 Ensemble Model Breakdown"):
                    for detail in ai_res["model_details"]:
                        st.write(f"**{detail['model']}**: AI={detail['ai_probability']:.1%} / Human={detail['human_probability']:.1%}")
                    st.caption(f"Ensemble: {ai_res.get('models_used', 1)} model(s) contributed")

            st.divider()

            st.subheader("Layer 1: Metadata & Provenance")
            if metadata_res.get("findings"):
                st.warning("Provenance indicators found!")
                st.json(metadata_res["findings"])
            else:
                st.info("No explicit AI provenance metadata found.")

            st.subheader("Layer 2: Invisible Watermark")
            if watermark_res.get("watermark_found"):
                st.warning(f"Watermark detected! Payload: {watermark_res.get('recovered_identifier')}")
            else:
                st.info("No recognizable watermark detected.")

            st.subheader("Layer 3: Perceptual Hashes")
            st.code(f"pHash: {hash_res.get('phash')}\ndHash: {hash_res.get('dhash')}\naHash: {hash_res.get('ahash')}")

            st.subheader("Layer 4: AI Model Detection (Ensemble)")
            if "error" in ai_res:
                st.error(f"Model Error: {ai_res['error']}")
            else:
                st.progress(ai_res.get("ai_probability", 0.0),
                            text=f"AI Probability: {ai_res.get('ai_probability', 0):.1%}")

            st.subheader("Layer 5: Image Forensics")

            st.markdown("**Error Level Analysis (ELA)** — detects tampering & compression anomalies")
            ela_sub = forensic_res.get("ela", {})
            if "error" in ela_sub:
                st.error(f"ELA Error: {ela_sub['error']}")
            else:
                st.write(f"ELA Anomaly: {ela_sub.get('anomaly_score', 0):.2f} (Variance: {ela_sub.get('variance', 0):.2f})")
                if "ela_image_base64" in forensic_res:
                    ela_img_bytes = base64.b64decode(forensic_res["ela_image_base64"])
                    st.image(ela_img_bytes, caption="ELA Heatmap", use_container_width=True)

            st.markdown("**Frequency Domain Analysis** — detects AI-generation spectral signatures")
            freq_sub = forensic_res.get("frequency", {})
            if "error" in freq_sub:
                st.error(f"Frequency Error: {freq_sub['error']}")
            else:
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric("HF Ratio", f"{freq_sub.get('hf_ratio', 0):.3f}")
                with fcol2:
                    st.metric("Spectral Entropy", f"{freq_sub.get('spectral_entropy', 0):.1f}")
                with fcol3:
                    st.metric("AI Frequency Score", f"{freq_sub.get('ai_frequency_score', 0):.2f}")
                if "spectrum_image_base64" in freq_sub:
                    spec_img_bytes = base64.b64decode(freq_sub["spectrum_image_base64"])
                    st.image(spec_img_bytes,
                             caption="Frequency Spectrum (bright centre = low freq, edges = high freq)",
                             use_container_width=True)

            st.markdown(f"**Composite Forensic Score**: {forensic_res.get('anomaly_score', 0):.2f} (30% ELA + 70% Frequency)")

        # ---- Feedback section (always visible after any scan) ----------------
        st.divider()
        st.subheader("🤖 Help Mimir Learn")
        st.write("Tell Mimir what this image **actually** is. This trains the detection weights immediately.")

        agent = get_rl_agent()
        buf = agent.get_buffer_status()
        st.caption(f"Pending buffered samples: {buf['buffered']}/{buf['needed']}  "
                   f"({buf['remaining']} more needed before next batch weight update)")

        fb_col1, fb_col2, fb_col3 = st.columns(3)

        def _apply_feedback(ground_truth: str):
            """Shared logic for AI / Human feedback buttons."""
            target = _LABEL_TO_TARGET[ground_truth]
            db.update_user_label(record_id, ground_truth)
            st.session_state["known_label"] = ground_truth

            features = st.session_state.get("last_features")
            if features is not None:
                # Correction → immediate single-sample update
                # Confirmation → also immediate (ground-truth is always high-value signal)
                agent.update_reward_immediate(features, target)
                st.session_state["last_features"] = None  # prevent double-submission

        with fb_col1:
            if st.button("🤖 This is AI", use_container_width=True, key="fb_ai"):
                _apply_feedback("AI")
                st.success("Recorded as **AI Generated**. Weights updated immediately. ✅")

        with fb_col2:
            if st.button("👤 This is Human", use_container_width=True, key="fb_human"):
                _apply_feedback("Human")
                st.success("Recorded as **Human Created**. Weights updated immediately. ✅")

        with fb_col3:
            if st.button("⏭️ Skip", use_container_width=True, key="fb_skip"):
                st.info("Feedback skipped.")

# ==============================================================================
with tab2:
    st.header("Registry Explorer")
    records = db.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records in registry yet.")

# ==============================================================================
with tab3:
    st.header("System Statistics")
    records = db.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.metric("Total Images Scanned", len(df))

        labeled = df[df["user_label"].notna()]
        st.metric("Ground-truth Labels Given", len(labeled))

        st.subheader("Assessments Breakdown")
        counts = df["final_assessment"].value_counts()
        st.bar_chart(counts)

        if len(labeled) > 0:
            st.subheader("User Label Breakdown")
            label_counts = labeled["user_label"].value_counts()
            st.bar_chart(label_counts)
    else:
        st.info("Scan some images to see statistics.")
