[README.md](https://github.com/user-attachments/files/31857768/README.md)
# SignSense AI 🤟

A real-time American Sign Language (ASL) gesture recognizer. It uses **MediaPipe** to track hand landmarks from a webcam feed and a **TensorFlow/Keras** neural network to classify the hand pose into a word or phrase, displayed live on screen with a confidence score.

## How it works

1. **Landmark extraction** — MediaPipe Hands detects 21 (x, y) landmarks on a single hand per frame.
2. **Normalization** — All landmarks are translated relative to the wrist (landmark 0), so the model is invariant to the hand's position in the frame. This produces a 42-value feature vector (21 landmarks × 2 coordinates).
3. **Classification** — A small fully-connected network (Dense 128 → Dropout → Dense 64 → Dense softmax) predicts the gesture class from the 42-feature vector.
4. **Live inference** — Predictions with confidence above 0.8 are overlaid on the webcam feed in real time.

## Recognized gestures

Trained on a custom-collected dataset of **1,892 samples across 8 classes**:

| Gesture | Samples |
|---|---|
| NO | 260 |
| FINE | 260 |
| EAT | 260 |
| THANK YOU | 253 |
| LIKE | 251 |
| HELLO | 250 |
| YES | 250 |
| HOW ARE YOU? | 108 |

Validation accuracy: _add your measured number here after running `train_model.py` — check the final epoch's `val_accuracy`._

## Tech stack

- **Computer Vision:** OpenCV, MediaPipe Hands
- **Deep Learning:** TensorFlow / Keras
- **Data handling:** Pandas, NumPy, scikit-learn (train/test split, label encoding)

## Project structure

```
SignSense_AI/
├── data/
│   └── sign_data.csv          # Collected landmark data (features + labels)
├── models/
│   ├── sign_model.keras       # Trained classifier
│   └── label_encoder.pkl      # Label encoder for class names
├── src/
│   ├── collect_data.py        # Webcam tool to record labeled gesture samples
│   ├── train_model.py         # Trains the classifier on collected data
│   └── app.py                 # Real-time webcam inference app
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Vedpatil01/SignSense_AI.git
cd SignSense_AI

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

**Run the live translator** (uses the pre-trained model included in `models/`):
```bash
python src/app.py
```
Press `q` to quit.

**Collect your own gesture data** (optional):
```bash
python src/collect_data.py
```
Follow the prompts to enter a label, position your hand in frame, and press **SPACEBAR** to capture each sample (300 recommended per gesture). Data is appended to `data/sign_data.csv`.

**Re-train the model** on your data:
```bash
python src/train_model.py
```
This regenerates `models/sign_model.keras` and `models/label_encoder.pkl`.

## Limitations

- Single-hand gestures only (`max_num_hands=1`); no support for two-handed signs.
- Recognizes static hand poses per frame rather than motion-based signs, so gestures that rely on movement (not just hand shape) aren't distinguished.
- Trained on a relatively small, self-collected dataset (~1,900 samples, 8 classes) — accuracy on new users' hand shapes/lighting may vary from validation numbers.

## Possible extensions

- Add motion-based gesture support (e.g. using a sequence model over landmark frames instead of single-frame classification).
- Expand the gesture vocabulary beyond the current 8 classes.
- Two-hand support for signs that require both hands.
