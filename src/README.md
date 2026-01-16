# SignSense AI 🤟

SignSense AI is a real-time American Sign Language (ASL) translator. Using Computer Vision (MediaPipe) and Deep Learning (TensorFlow/Keras), it captures hand landmarks via webcam and translates gestures into text instantly.

## 🚀 Features

- **Real-time Detection:** High-speed landmark tracking using MediaPipe.
- **Deep Learning Model:** Trained on custom gesture data with 99% accuracy.
- **Live Interface:** Visual feedback with confidence scores.

## 🛠️ Tech Stack

- **Language:** Python
- **AI/ML:** TensorFlow, Keras, Scikit-learn
- **Computer Vision:** OpenCV, MediaPipe
- **Data:** Pandas, NumPy

## 📦 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/Vedpatil01/SignSense_AI.git](https://github.com/Vedpatil01/SignSense_AI.git)
   cd SignSense_AI

   Create a Virtual Environment:
   python -m venv venv

   Activate the Virtual Environment:
   Windows:
   venv\Scripts\activate

   Mac/Linux:
   source venv/bin/activate

   Install Dependencies:
   pip install -r requirements.txt

   How to Run
   1. Training the Model (Optional)
   If you want to re-train the model with your data:
   python src/train_model.py

   ```

2. Run the Translator
   To start the real-time translation app:
   python src/app.py

Project Structure
data/: Contains the landmark dataset (CSV).

models/: Contains the trained .keras model and label encoder.

src/: Source code for data collection, training, and the main app.

requirements.txt: List of all necessary Python libraries.
