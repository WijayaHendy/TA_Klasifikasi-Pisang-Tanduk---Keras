import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# 1. Muat Model Anda
model = load_model('Citra Pisang Tanduk_Xception-Pisang Tanduk Xception-93.06.keras')

# 2. Siapkan Folder Test (Pastikan Anda punya folder 'dataset/test' yang berisi folder tiap kelas)
test_dir = r'D:\TA_Klasifikasi Pisang Tanduk - Keras\dataset\train'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3. Prediksi Semua Gambar di Folder Test
print("Sedang menghitung metrik... Mohon tunggu.")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
target_names = list(test_generator.class_indices.keys())

# 4. TAMPILKAN HASILNYA
print("\n========== HASIL EVALUASI MODEL ==========")
print(classification_report(y_true, y_pred, target_names=target_names))