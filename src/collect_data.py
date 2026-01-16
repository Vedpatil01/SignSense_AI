"""
Professional Sign Language Data Collection Script
Senior ML Engineer Implementation
Normalizes 21 hand landmarks (x,y) relative to wrist (landmark 0) for distance-invariant features
Captures 300 samples per gesture class into consolidated CSV
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path

class SignLanguageDataCollector:
    def __init__(self):
        # Initialize MediaPipe Hands with high confidence for production quality
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        
        # Create data directory if it doesn't exist
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.csv_path = self.data_dir / 'sign_data.csv'
        
        # Collection tracking
        self.collected_data = []
        self.current_label = ""
        self.samples_needed = 300
        self.samples_collected = 0
        
        print("=== Sign Language Data Collector Initialized ===")
        print(f"Target: {self.samples_needed} samples per gesture")
        print(f"Output: {self.csv_path}")
    
    def normalize_landmarks(self, landmarks):
        """Normalize all 21 landmarks relative to wrist (landmark 0)"""
        # Extract x,y coordinates (21x2 array)
        coords = np.array([[lm.x, lm.y] for lm in landmarks])
        
        # Wrist is landmark 0 - normalize all points relative to wrist
        wrist = coords[0]
        normalized = coords - wrist
        
        # Flatten to 42-feature vector (21 landmarks * 2 coords)
        return normalized.flatten()
    
    def collection_countdown(self, frame, seconds=3):
        """Display professional 3-second countdown overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        for i in range(seconds, 0, -1):
            status = f"HOLD GESTURE - Starting in {i}"
            cv2.putText(frame, status, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Sign Language Data Collection', frame)
            cv2.waitKey(1000)
    
    def collect_samples(self, label):
        """Main collection loop for specified label"""
        print(f"\n🎯 Collecting {self.samples_needed} samples for '{label}'")
        print("📹 Position hand clearly in frame. Press SPACEBAR after countdown.")
        
        self.current_label = label
        self.samples_collected = 0
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.samples_collected < self.samples_needed:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Status overlay
            cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
            cv2.putText(frame, f"Label: {label} | Collected: {self.samples_collected}/{self.samples_needed}", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            if results.multi_hand_landmarks:
                hlm = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hlm, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract and normalize landmarks
                normalized_features = self.normalize_landmarks(hlm.landmark)
                
                status_text = "✓ FEATURES READY - Press SPACEBAR to CAPTURE"
                color = (0, 255, 0)
            else:
                normalized_features = np.zeros(42)  # Empty feature vector
                status_text = "✗ No hand detected - Position hand clearly"
                color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Sign Language Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar to capture
                if results.multi_hand_landmarks:  # Only save valid detections
                    self.collected_data.append({
                        'features': normalized_features,
                        'label': label
                    })
                    self.samples_collected += 1
                    print(f"✅ Saved sample {self.samples_collected}/{self.samples_needed}")
                    
                    # Quick feedback flash
                    cv2.putText(frame, "SAVED!", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow('Sign Language Data Collection', frame)
                    cv2.waitKey(200)
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_data(self):
        """Save all collected data to CSV with professional formatting"""
        if not self.collected_data:
            print("❌ No data collected!")
            return
        
        # Convert features to string format for CSV storage
        df = pd.DataFrame(self.collected_data)
        df['features'] = df['features'].apply(lambda x: ' '.join(map(str, x)))
        
        # Append to existing CSV or create new
        if self.csv_path.exists():
            existing_df = pd.read_csv(self.csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(self.csv_path, index=False)
        print(f"\n💾 Data saved to {self.csv_path}")
        print(f"📊 Total samples: {len(df)} across {df['label'].nunique()} classes")
        print("\n📈 Class distribution:")
        print(df['label'].value_counts().sort_index())

def main():
    collector = SignLanguageDataCollector()
    
    print("\n" + "="*60)
    print("🎮 SIGN LANGUAGE DATA COLLECTION")
    print("="*60)
    print("Instructions:")
    print("1. Run script and enter gesture label (A-Z, Hello, ThankYou, etc.)")
    print("2. After entering label, position hand clearly in frame")
    print("3. Wait for 3-second countdown, then Press SPACEBAR 300 times")
    print("4. Press 'q' anytime to quit collection for that label")
    print("5. Repeat for different gestures")
    print("="*60)
    
    try:
        while True:
            label = input("\nEnter gesture label (or 'quit' to finish): ").strip().upper()
            if label.lower() == 'quit':
                break
            
            if not label:
                print("⚠️ Please enter a valid label")
                continue
            
            collector.collect_samples(label)
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        collector.save_data()
        print("\n🎉 Data collection complete!")
        print(f"📁 Final dataset: {collector.csv_path}")
        print("Next: Run training script with this data!")

if __name__ == "__main__":
    main()
