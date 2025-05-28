import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

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

# Evaluasi
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")

# Simpan model
joblib.dump(model, 'model/model_svm.pkl')
joblib.dump(scaler, 'model/scaler.pkl')