import streamlit as st
import pandas as pd
import io
import concurrent.futures
import base64

from detectors.metadata import analyze_metadata
from detectors.watermark import detect_watermark
from detectors.hashing import generate_hashes
from detectors.ai_model import get_ai_detector
from forensics.ela import error_level_analysis
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
                    f_ela = executor.submit(error_level_analysis, io.BytesIO(file_bytes))
                    
                    metadata_res = f_meta.result()
                    watermark_res = f_water.result()
                    hash_res = f_hash.result()
                    ai_res = f_ai.result()
                    ela_res = f_ela.result()
                
                final_score, label, conf = calculate_final_assessment(
                    metadata_res, watermark_res, ai_res, ela_res
                )
                
                db.add_record(
                    filename=uploaded_file.name,
                    phash=hash_res.get("phash"),
                    dhash=hash_res.get("dhash"),
                    ahash=hash_res.get("ahash"),
                    watermark_id=watermark_res.get("recovered_identifier"),
                    metadata=metadata_res.get("findings"),
                    ai_score=ai_res.get("ai_probability", 0),
                    human_score=ai_res.get("human_probability", 0),
                    final_assessment=label
                )

                # Store features in session state for the feedback loop
                agent = get_rl_agent()
                st.session_state["last_features"] = agent.extract_features(
                    metadata_res, watermark_res, ai_res, ela_res
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
            st.code(f"pHash: {hash_res.get('phash')}\\ndHash: {hash_res.get('dhash')}\\naHash: {hash_res.get('ahash')}")
            
            st.subheader("Layer 4: AI Model Detection")
            if "error" in ai_res:
                st.error(f"Model Error: {ai_res['error']}")
            else:
                st.progress(ai_res.get("ai_probability", 0.0), text=f"AI Probability: {ai_res.get('ai_probability', 0):.1%}")
                
            st.subheader("Layer 5: Image Forensics (ELA)")
            if "error" in ela_res:
                st.error(f"Forensic Error: {ela_res['error']}")
            else:
                st.write(f"Anomaly Score: {ela_res.get('anomaly_score', 0):.2f} (Variance: {ela_res.get('variance', 0):.2f})")
                if "ela_image_base64" in ela_res:
                    ela_img_bytes = base64.b64decode(ela_res["ela_image_base64"])
                    st.image(ela_img_bytes, caption="Error Level Analysis (ELA) Heatmap", width="stretch")

            # RL Feedback Loop
            st.divider()
            st.subheader("🤖 Help Mimir Learn")
            st.write("Was this prediction correct? Your feedback trains the detection weights in real-time.")
            col_yes, col_no, col_skip = st.columns(3)
            with col_yes:
                if st.button("✅ Yes — Correct", use_container_width=True):
                    if "last_features" in st.session_state:
                        agent = get_rl_agent()
                        # Reward: the current label was right, so reinforce it
                        is_ai = 1.0 if label in ["Likely AI Generated", "Possible AI Generated", "Verified Provenance Match (AI)", "Verified Watermark Match (AI)"] else 0.0
                        agent.update_reward(st.session_state["last_features"], is_ai)
                        st.success("Thanks! Mimir's weights have been updated. 🧠")
            with col_no:
                if st.button("❌ No — Incorrect", use_container_width=True):
                    if "last_features" in st.session_state:
                        agent = get_rl_agent()
                        # Penalize: flip the target
                        is_ai = 0.0 if label in ["Likely AI Generated", "Possible AI Generated", "Verified Provenance Match (AI)", "Verified Watermark Match (AI)"] else 1.0
                        agent.update_reward(st.session_state["last_features"], is_ai)
                        st.warning("Noted! Mimir will adjust its weights accordingly. 🔄")
            with col_skip:
                if st.button("⏭️ Skip", use_container_width=True):
                    st.info("Feedback skipped.")

with tab2:
    st.header("Registry Explorer")
    records = db.get_all_records()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, width="stretch")
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
