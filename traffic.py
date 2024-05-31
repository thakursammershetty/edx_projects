import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def load_data(data_dir):
    images = []
    labels = []
    
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            img = cv2.imread(filepath)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)
    
    return (np.array(images), np.array(labels))

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    import sys
    
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")
    
    data_dir = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    images, labels = load_data(data_dir)
    labels = tf.keras.utils.to_categorical(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE)
    
    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    
    model.evaluate(x_test, y_test, verbose=2)
    
    if model_file:
        model.save(model_file)

if __name__ == "__main__":
    main()

