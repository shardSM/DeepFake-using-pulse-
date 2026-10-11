"""PulseCheck: Deepfake Video Detector via Remote Photoplethysmography (rPPG).

Streamlit Web Dashboard implementing Intel's FakeCatcher biological approach.
"""

from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st

import predict

st.set_page_config(
    page_title="PulseCheck — rPPG Deepfake Detector",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 PulseCheck: Biological Deepfake Detector")

st.markdown(
    """
    **PulseCheck** detects synthetic portrait videos using **remote photoplethysmography (rPPG)**.
    Real human faces exhibit subtle, periodic skin-color changes caused by blood flow (hemoglobin absorbs green light).
    These pulse signals are spatially coherent across the forehead, cheeks, and chin. Because deepfake generators
    synthesize pixels without an underlying physiological cardiovascular model, their pulse signals are weak,
    noisy, or inconsistent across facial regions.
    """
)

CKPT_PATH = "checkpoints/best.pt"

# Checkpoint presence check
if not Path(CKPT_PATH).exists():
    st.error(
        "⚠️ **Train a model first!** No model checkpoint found at `checkpoints/best.pt`. "
        "Please build a dataset with `build_dataset.py` and run `train.py` before analyzing videos."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a portrait video (MP4, MOV, AVI, MKV)",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a video featuring a visible human face with minimal extreme head rotations.",
)

if uploaded_file is not None:
    # Save uploaded bytes to temporary file for analysis
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_video_path = tmp.name

    try:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Video")
            st.video(tmp_video_path)

        with col2:
            st.subheader("Analysis")
            with st.spinner("Extracting face landmarks, rPPG signals, and evaluating PPG maps..."):
                results = predict.analyse(tmp_video_path, ckpt=CKPT_PATH)

            verdict = results["verdict"]

            if verdict == "NO_SIGNAL":
                st.warning(f"⚠️ **NO SIGNAL DETECTED**: {results.get('reason')}")
            else:
                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)

                if verdict == "REAL":
                    m1.metric("Verdict", "✅ REAL", delta="Authentic pulse")
                else:
                    m1.metric("Verdict", "🚨 FAKE", delta="-Incoherent pulse", delta_color="inverse")

                m2.metric("Fake Score", f"{results['fake_score']:.3f}", f"Threshold: {results['threshold']:.2f}")

                hr_val = f"{results['heart_rate_bpm']:.1f} BPM" if results["heart_rate_bpm"] else "N/A"
                m3.metric("Heart Rate", hr_val)

                m4.metric("Pulse Coherence", f"{results['pulse_coherence']:.3f}")

                # Timeline Chart
                if results["timeline"]:
                    st.subheader("Temporal Fake Probability Timeline")
                    df = pd.DataFrame(results["timeline"])
                    df["threshold"] = results["threshold"]
                    df = df.rename(columns={"t": "Time (s)", "p_fake": "Fake Probability", "threshold": "Decision Threshold"})
                    df = df.set_index("Time (s)")
                    st.line_chart(df)

    finally:
        Path(tmp_video_path).unlink(missing_ok=True)

with st.expander("ℹ️ How it works: Biological FakeCatcher Approach"):
    st.markdown(
        """
        1. **Facial Tracking**: MediaPipe FaceMesh tracks 468 landmark coordinates per frame.
        2. **32 Adaptive Patches**: Extracts skin patches across forehead, cheeks, nose, and chin scaled to facial width.
        3. **Uniform Resampling**: Interpolates tracking traces to a constant 30 FPS.
        4. **POS rPPG Algorithm**: The Plane-Orthogonal-to-Skin algorithm eliminates lighting noise and projects chromatic signals.
        5. **PPG Maps**: Generates (3 channels × 32 ROIs × 128 frames) spatio-temporal representations.
        6. **PPGNet CNN**: Classifies window maps and aggregates probabilities to predict video authenticity.
        """
    )

st.caption(
    "⚠️ **Limitations**: Heavily compressed video files, poor illumination, or rapid head motion can degrade biological signals. "
    "PulseCheck is designed as an investigative research tool and is not intended for standalone courtroom forensics."
)
