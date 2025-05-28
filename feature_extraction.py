import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
import joblib

def extract_features(image):
    # Pra-pemrosesan
    img = cv2.resize(image, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ekstraksi fitur warna (RGB dan HSV)
    color_features = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        color_features.extend(hist.flatten())
    
    # Ekstraksi fitur tekstur (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    # Gabungkan semua fitur
    features = color_features + [contrast, dissimilarity, homogeneity, energy, correlation]
    return features

def save_features(dataset_path):
    features = []
    labels = []
    classes = ['dara', 'ijo', 'simping']
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                feature_vector = extract_features(image)
                features.append(feature_vector)
                labels.append(class_name)
    
    # Simpan fitur
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv('features.csv', index=False)
    return df

if __name__ == "__main__":
    dataset_path = 'dataset'
    save_features(dataset_path)