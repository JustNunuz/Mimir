import streamlit as st
import pandas as pd
import io
import concurrent.futures
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

@st.cache_resource
def load_detector():
    return get_ai_detector()

detector = load_detector()

st.title("👁️ Mimir: AI Image Provenance & Detection")
st.markdown("Analyze images using a multi-layered approach to detect AI generation and provenance.")

tab1, tab2, tab3 = st.tabs(["🔍 Scan Image", "📚 Registry Explorer", "📊 System Statistics"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image to analyze...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Run Full Scan"):
            with st.spinner("Analyzing Layers in Parallel..."):
                file_bytes = uploaded_file.getvalue()

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    f_meta = executor.submit(analyze_metadata, io.BytesIO(file_bytes))
                    f_water = executor.submit(detect_watermark, io.BytesIO(file_bytes))
                    f_hash = executor.submit(generate_hashes, io.BytesIO(file_bytes))
                    f_ai = executor.submit(detector.predict, io.BytesIO(file_bytes))
                    f_forensic = executor.submit(combined_forensic_analysis, io.BytesIO(file_bytes))

                    metadata_res = f_meta.result()
                    watermark_res = f_water.result()
                    hash_res = f_hash.result()
                    ai_res = f_ai.result()
                    forensic_res = f_forensic.result()

                final_score, label, conf = calculate_final_assessment(
                    metadata_res, watermark_res, ai_res, forensic_res
                )

                db.add_record(
                    filename=uploaded_file.name,
                    phash=hash_res.get("phash"),
                    dhash=hash_res.get("dhash"),
                    ahash=hash_res.get("ahash"),
                    watermark_id=watermark_res.get("recovered_identifier"),
                    metadata=metadata_res.get("findings"),
                    # Store final_score (post-RL-blend, post-floor) not the raw ensemble output,
                    # so the registry matches what the user actually saw in the UI.
                    ai_score=final_score,
                    human_score=1.0 - final_score,
                    final_assessment=label
                )

                # Store features in session state for the feedback loop
                agent = get_rl_agent()
                st.session_state["last_features"] = agent.extract_features(
                    metadata_res, watermark_res, ai_res, forensic_res
                )
                st.session_state["scan_complete"] = True

            st.success("Scan Complete!")

            st.header("Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Assessment", label)
            with col2:
                st.metric("AI Probability", f"{final_score:.1%}")
            with col3:
                st.metric("Confidence", f"{conf:.1%}")

            # Show ensemble details if available
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
                st.progress(ai_res.get("ai_probability", 0.0), text=f"AI Probability: {ai_res.get('ai_probability', 0):.1%}")

            st.subheader("Layer 5: Image Forensics")

            # ELA sub-section
            st.markdown("**Error Level Analysis (ELA)** — detects tampering & compression anomalies")
            ela_sub = forensic_res.get("ela", {})
            if "error" in ela_sub:
                st.error(f"ELA Error: {ela_sub['error']}")
            else:
                st.write(f"ELA Anomaly: {ela_sub.get('anomaly_score', 0):.2f} (Variance: {ela_sub.get('variance', 0):.2f})")
                if "ela_image_base64" in forensic_res:
                    ela_img_bytes = base64.b64decode(forensic_res["ela_image_base64"])
                    st.image(ela_img_bytes, caption="ELA Heatmap", use_container_width=True)

            # Frequency analysis sub-section
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
                    st.image(spec_img_bytes, caption="Frequency Spectrum (bright centre = low freq, edges = high freq)", use_container_width=True)

            st.markdown(f"**Composite Forensic Score**: {forensic_res.get('anomaly_score', 0):.2f} (30% ELA + 70% Frequency)")

            # RL Feedback Loop
            st.divider()
            st.subheader("🤖 Help Mimir Learn")
            st.write("Was this prediction correct? Your feedback trains the detection weights.")

            # Show buffer status
            agent = get_rl_agent()
            buf = agent.get_buffer_status()
            st.caption(f"Feedback buffer: {buf['buffered']}/{buf['needed']} samples "
                       f"({buf['remaining']} more needed before next weight update)")

            # Soft reward targets scaled by label tier.
            # Flat binary targets erase the tier information the scoring system expresses,
            # and Inconclusive confirmations carry no meaningful ground-truth signal.
            _SOFT_TARGET = {
                "Verified Provenance Match (AI)":  0.95,
                "Verified Watermark Match (AI)":   0.95,
                "Likely AI Generated":             0.90,
                "Possible AI Generated":           0.70,
                "Inconclusive":                    None,   # excluded from training
                "Possible Human Created":          0.30,
                "Likely Human Created":            0.10,
            }

            col_yes, col_no, col_skip = st.columns(3)
            with col_yes:
                if st.button("✅ Yes — Correct", use_container_width=True):
                    if "last_features" in st.session_state:
                        target = _SOFT_TARGET.get(label)
                        if target is None:
                            st.info("Inconclusive results are excluded from training — no signal to learn from.")
                        else:
                            agent = get_rl_agent()
                            updated = agent.update_reward(st.session_state["last_features"], target)
                            if updated:
                                st.success("Batch threshold reached — weights updated! 🧠")
                            else:
                                st.info(f"Feedback recorded. {agent.get_buffer_status()['remaining']} more needed.")
            with col_no:
                if st.button("❌ No — Incorrect", use_container_width=True):
                    if "last_features" in st.session_state:
                        correct_target = _SOFT_TARGET.get(label)
                        if correct_target is None:
                            st.info("Inconclusive results are excluded from training — no signal to learn from.")
                        else:
                            # Flip: if model said AI, user says human, and vice-versa
                            flipped_target = 1.0 - correct_target
                            agent = get_rl_agent()
                            updated = agent.update_reward(st.session_state["last_features"], flipped_target)
                            if updated:
                                st.warning("Batch threshold reached — weights adjusted! 🔄")
                            else:
                                st.info(f"Feedback recorded. {agent.get_buffer_status()['remaining']} more needed.")
            with col_skip:
                if st.button("⏭️ Skip", use_container_width=True):
                    st.info("Feedback skipped.")

with tab2:
    st.header("Registry Explorer")
    records = db.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No records in registry yet.")

with tab3:
    st.header("System Statistics")
    records = db.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.metric("Total Images Scanned", len(df))

        st.subheader("Assessments Breakdown")
        counts = df['final_assessment'].value_counts()
        st.bar_chart(counts)
    else:
        st.info("Scan some images to see statistics.")
