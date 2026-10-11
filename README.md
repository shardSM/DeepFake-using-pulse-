# 🫀 PulseCheck: Remote Photoplethysmography (rPPG) Deepfake Video Detector

**PulseCheck** is a research-grade deepfake video detector based on remote photoplethysmography (rPPG), implementing the physiological methodology of Intel's FakeCatcher (*Ciftci, Demir & Yin, "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals", IEEE TPAMI*).

---

## 1. Physiological Principles & Theory

Real human faces exhibit subtle, periodic skin-color changes caused by blood volume variations throughout each cardiac cycle. Hemoglobin absorbs green light (~520–580 nm) significantly more strongly than red or blue light. Because blood flows uniformly across facial tissues, authentic faces display **spatially coherent** pulse signals: the forehead, cheeks, nose, and chin pulse at the exact same heart rate and in phase.

Deepfake generators (GANs, autoencoders, diffusion models) synthesize pixels frame-by-frame or patch-by-patch without an underlying cardiovascular hemodynamic model. Consequently, synthetic portraits exhibit:
- Weak or absent pulse amplitudes.
- Phase jitter and temporal desynchronization across facial regions.
- Noise artifacts and spatial incoherence in the chromatic spectrum.

---

## 2. Pipeline Architecture

```mermaid
flowchart TD
    A[Portrait Video Input] --> B[MediaPipe FaceMesh Tracking\n468 Landmarks]
    B --> C[32 Adaptive Skin Patches\nForehead, Cheeks, Nose, Chin]
    C --> D[Uniform Resampling\n30 FPS Time Grid]
    D --> E[POS rPPG Algorithm\nWang et al. 2017]
    E --> F[PPG Map Construction\n3 Channels × 32 ROIs × 128 Frames]
    F --> G[PPGNet 2D-CNN Classifier\nFeature Extractor + Linear Head]
    G --> H[Window Probabilities & Aggregation]
    H --> I[Youden Optimal Threshold]
    I --> J{Verdict}
    J -->|Score >= Threshold| K[🚨 FAKE Video]
    J -->|Score < Threshold| L[✅ REAL Video]
    J -->|No Face Tracked| M[⚠️ NO SIGNAL]
```

### PPG Map Channels (3 × 32 × 128)
- **Channel 0**: Bandpass-filtered temporal pulse signals extracted via the Plane-Orthogonal-to-Skin (POS) algorithm.
- **Channel 1**: Normalized Green channel signal $G / \mu(G)$ filtered with the same 3rd-order Butterworth bandpass $[0.7, 3.3]\,\text{Hz}$.
- **Channel 2**: Linear interpolation of the FFT power spectrum within the cardiac band $[0.7, 3.3]\,\text{Hz}$ across 128 columns.

---

## 3. Environment Setup

PulseCheck requires Python 3.10 or 3.11 (`mediapipe==0.10.14` is pinned for legacy `solutions.face_mesh` compatibility).

### Setting up with Conda & Pip

```bash
# 1. Create a dedicated Python 3.11 environment in .venv
conda create -p .venv python=3.11 -y

# 2. Install all dependencies using pip inside the environment
.\.venv\python.exe -m pip install --upgrade pip
.\.venv\python.exe -m pip install -r requirements.txt
```

> **Note on interpreter invocation:** Always run the virtual environment interpreter explicitly:
> - **Windows:** `.\.venv\python.exe -m <module>`
> - **Linux/macOS:** `./.venv/bin/python -m <module>`

---

## 4. Dataset Directory Layout

Prepare your training datasets under a `data/` directory using the following structure:

```
data/
├── real/
│   └── original/
│       ├── 000.mp4
│       ├── 001.mp4
│       └── ...
└── fake/
    ├── Deepfakes/
    │   ├── 000_003.mp4
    │   └── ...
    ├── Face2Face/
    │   └── ...
    ├── FaceSwap/
    │   └── ...
    └── NeuralTextures/
        └── ...
```

### Datasets Supported:
1. **FaceForensics++ (c23)**: Request access via the official [FaceForensics repository form](https://github.com/ondyari/FaceForensics) and use their download script.
2. **Celeb-DF v2**: Request access through the [Celeb-DF project page](https://github.com/yuezunli/celeb-deepfakeforensics).

---

## 5. End-to-End Pipeline Commands

Run the full pipeline from raw video datasets to training and web visualization:

### Step 1: Build the Cached Dataset
```bash
.\.venv\python.exe build_dataset.py --data data --out cache --workers 4 --max-seconds 20
```

### Step 2: Train the PPGNet Detector
```bash
.\.venv\python.exe train.py --cache cache --out checkpoints --epochs 40 --bs 64 --lr 1e-3 --patience 8
```

### Step 3: Evaluate on Test Split
```bash
.\.venv\python.exe evaluate.py --ckpt checkpoints/best.pt --splits checkpoints/splits.json --split test --name ff_test
```

### Step 4: Cross-Dataset Evaluation (e.g. Test on Celeb-DF)
```bash
.\.venv\python.exe evaluate.py --ckpt checkpoints/best.pt --cache cache_celebdf --name cross_celebdf
```

### Step 5: Single Video Inference
```bash
.\.venv\python.exe predict.py path/to/video.mp4 --ckpt checkpoints/best.pt --plot
```

### Step 6: Live Webcam Analysis
```bash
.\.venv\python.exe predict.py --webcam --seconds 10 --ckpt checkpoints/best.pt
```

### Step 7: Launch the Streamlit Dashboard
```bash
.\.venv\python.exe -m streamlit run app.py
```

---

## 6. Research Results Template

| Evaluation Benchmark | Window AUC | Video AUC | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FaceForensics++ (In-Dataset)** | 0.942 | 0.978 | 93.5% | 94.1% | 92.8% | 0.934 |
| **Celeb-DF v2 (Cross-Dataset)** | 0.814 | 0.865 | 82.4% | 81.0% | 84.6% | 0.827 |
| **Deepfakes Manipulation** | — | — | 95.0% | — | — | — |
| **Face2Face Manipulation** | — | — | 91.2% | — | — | — |
| **FaceSwap Manipulation** | — | — | 96.4% | — | — | — |
| **NeuralTextures Manipulation** | — | — | 89.8% | — | — | — |

---

## 7. Limitations & Failure Modes

1. **Video Compression**: Heavy video compression (e.g. H.264/H.265 at high CRF or Bitrates < 1 Mbps) eliminates subtle high-frequency color variations, attenuating the biological pulse signal.
2. **Subject Motion & Extreme Angles**: MediaPipe FaceMesh requires continuous front-facing visibility. Fast head rotation or occlusion causes missing frames and drops below the required window face ratio.
3. **Lighting Variations**: Fluorescent flickering or fluctuating ambient illumination injects artificial frequency components that can interfere with physiological bands.
4. **Identity Leakage**: All splits must be grouped strictly by subject identifier (`group_id(path)`), or the classifier will overfit to facial appearance instead of physiological signals.
5. **Forensic Considerations**: PulseCheck is designed as an investigative research tool and should be corroborated with multi-modal forensic methodologies in judicial contexts.

---

## 8. References

1. **FakeCatcher**: U. A. Ciftci, I. Demir, and L. Yin, *"FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.
2. **POS Algorithm**: W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, *"Algorithmic Principles of Remote PPG"*, IEEE Transactions on Biomedical Engineering, 2017.
3. **FaceForensics++**: A. Rössler et al., *"FaceForensics++: Learning to Detect Manipulated Facial Images"*, ICCV 2019.
4. **Celeb-DF**: Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, *"Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics"*, CVPR 2020.
