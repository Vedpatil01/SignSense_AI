import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Standard TensorFlow import
import tensorflow as tf

def train_sign_model():
    data_path = 'data/sign_data.csv'
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Convert features to proper float32 arrays (crucial for TF 2.x)
    X = np.array([np.fromstring(f, sep=' ') for f in df['features']]).astype('float32')
    y = df['label']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype('int32')
    num_classes = len(label_encoder.classes_)
    
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 2. Sequential model using input_shape inside the first Dense layer
    # This avoids the 'batch_shape' conflict between different Keras versions
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    print(f"🚀 Training starting for {num_classes} classes...")
    
    # Run training
    model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=32, 
              validation_data=(X_test, y_test),
              verbose=1)

    # 3. Save as .keras (modern format)
    model.save(os.path.join(model_dir, 'sign_model.keras'))
    print(f"✅ Success! Model saved in the '{model_dir}' folder.")

if __name__ == "__main__":
    train_sign_model()