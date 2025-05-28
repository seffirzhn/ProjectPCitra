import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset fitur
df = pd.read_csv('features.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Latih model SVM
model = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['dara', 'ijo', 'simping'], output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Simpan metrik evaluasi
evaluation_metrics = {
    'accuracy': accuracy,
    'report': report,
    'confusion_matrix': conf_matrix.tolist()  # Konversi numpy array ke list
}

# Simpan model dan metrik
joblib.dump(model, 'model/model_svm.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(evaluation_metrics, 'model/evaluation_metrics.pkl')

# Cetak laporan
print(f"Akurasi: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['dara', 'ijo', 'simping']))

# Visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['dara', 'ijo', 'simping'],
            yticklabels=['dara', 'ijo', 'simping'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('model/confusion_matrix.png')
plt.close()