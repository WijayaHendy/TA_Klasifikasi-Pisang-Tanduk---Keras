# train_model.py - versi menggunakan MobileNetV2 + fine-tuning + augmentasi lanjutan

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Path dataset
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Image preprocessing
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.8,1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.2
)

train_data = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_data = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Load MobileNetV2 tanpa top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Tambahkan layer klasifikasi baru
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Hitung class weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes)
class_weights_dict = dict(enumerate(class_weights))

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('model_best.h5', monitor='val_accuracy', save_best_only=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training awal
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stop, checkpoint, lr_schedule]
)

# Fine-tuning: buka 10 layer terakhir
for layer in base_model.layers[-10:]:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Training lanjutan
history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_schedule]
)

# Simpan model final
model.save('model_pisang.h5')

# Grafik Akurasi dan Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'] + history_ft.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'] + history_ft.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Selama Pelatihan')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'] + history_ft.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'] + history_ft.history['val_loss'], label='Loss Validasi')
plt.title('Loss Selama Pelatihan')
plt.legend()

plt.savefig('grafik_training.png')
plt.close()

# Evaluasi akhir
val_preds = model.predict(val_data)
y_pred = np.argmax(val_preds, axis=1)
y_true = val_data.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.figure(figsize=(8,6))
ConfusionMatrixDisplay(cm, display_labels=list(val_data.class_indices.keys())).plot(cmap='Blues')

# ... (kode training, plotting, dan evaluasi Anda sebelumnya) ...

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# --- TAMBAHAN BARU: Konversi Model ke TensorFlow Lite ---

print("\n--- Memulai Konversi ke TFLite ---")

# Muat model terbaik yang disimpan oleh ModelCheckpoint
try:
    best_model = tf.keras.models.load_model('model_best.h5')
    print("Model terbaik (model_best.h5) berhasil dimuat.")

    # Inisialisasi TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    
    # Aktifkan optimisasi (opsional, tapi disarankan)
    # Ini akan mengurangi ukuran file dan latensi, dengan sedikit penurunan akurasi
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Lakukan konversi
    tflite_model = converter.convert()
    
    # Simpan model TFLite ke file
    tflite_model_path = 'model_pisang.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\nKonversi berhasil!")
    print(f"Model TFLite telah disimpan di: {tflite_model_path}")
    print(f"Ukuran file H5: {os.path.getsize('model_best.h5') / 1024:.2f} KB")
    print(f"Ukuran file TFLite: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")

except Exception as e:
    print(f"\nTerjadi kesalahan saat konversi TFLite: {e}")