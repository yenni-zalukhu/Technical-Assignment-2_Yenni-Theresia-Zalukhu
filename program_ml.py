import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Membaca Dataset
data_path = 'ai4i2020.csv'  # Sesuaikan path file dengan benar
df = pd.read_csv(data_path)

# 2. Eksplorasi Data
print("Informasi Data:")
print(df.info())
print("\nStatistik Deskriptif:")
print(df.describe())

# 3. Pembersihan Data
# Cek missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Hapus kolom yang tidak diperlukan atau yang mengandung string
df = df.drop(columns=['UDI', 'Product ID'])

# Encode categorical variables jika ada
# Misalnya kolom 'Type' harus diubah ke numerik jika perlu
df['Type'] = df['Type'].astype('category').cat.codes

# 4. Pra-pemrosesan Data
# Pisahkan fitur dan target
X = df.drop(columns=['TWF'])  # Ganti dengan nama kolom target yang sesuai
y = df['TWF']  # Ganti dengan nama kolom target yang sesuai

# Pembagian data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Pembangunan Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

# 7. Visualisasi Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 8. Prediksi
# Fungsi untuk prediksi baru
def predict_tool_wear_failure(new_data):
    # Konversi new_data menjadi DataFrame untuk memudahkan validasi fitur
    new_data_df = pd.DataFrame(new_data, columns=X.columns)
    # Pastikan fitur baru memiliki nama dan urutan yang sama dengan data yang dilatih
    new_data_scaled = scaler.transform(new_data_df)
    prediction = model.predict(new_data_scaled)
    return prediction

# Contoh data baru (pastikan urutannya sesuai dengan X_train)
# Menggunakan kolom yang sama dan dalam urutan yang benar
data_baru = {
    'Type': [1],  # Misalnya 1 adalah encoding untuk 'Type' tertentu
    'Air temperature [K]': [298.1],
    'Process temperature [K]': [308.2],
    'Rotational speed [rpm]': [1550],
    'Torque [Nm]': [42.5],
    'Tool wear [min]': [400],
    'Machine failure': [0],  # Nilai dummy, tidak digunakan untuk prediksi
    'HDF': [0],  # Nilai dummy, tidak digunakan untuk prediksi
    'PWF': [0],  # Nilai dummy, tidak digunakan untuk prediksi
    'OSF': [0],  # Nilai dummy, tidak digunakan untuk prediksi
    'RNF': [0]   # Nilai dummy, tidak digunakan untuk prediksi
}

# Konversi data_baru menjadi DataFrame
data_baru_df = pd.DataFrame(data_baru)

# Pastikan jumlah kolom sesuai dengan data latih
print("Data baru untuk prediksi:")
print(data_baru_df)

# Lakukan prediksi
prediksi = predict_tool_wear_failure(data_baru_df)
print("Prediksi kegagalan keausan pahat:", prediksi)
