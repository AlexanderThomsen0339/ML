import os
import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 1. Indstillinger
# ===============================
dataset_dir = "raw"
image_size = (48, 48)    
classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
num_classes = len(classes)

# ===============================
# 2. Initialiser ansigtsdetektor
# ===============================
detector = MTCNN()

# ===============================
# 3. Funktion til at preprocess billeder
# ===============================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face, image_size)
        face_normalized = face_resized / 255.0
        return face_normalized
    else:
        return None

# ===============================
# 4. Load billeder og labels
# ===============================
X = []
y = []

for idx, label in enumerate(classes):
    folder = os.path.join(dataset_dir, label)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        face = preprocess_image(file_path)
        if face is not None:
            X.append(face)
            y.append(idx)

X = np.array(X)
y = np.array(y)

# One-hot encode labels
y_onehot = to_categorical(y, num_classes=num_classes)

# ===============================
# 5. Split i train/validation/test
# ===============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_onehot, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1), random_state=42
)

# ===============================
# 6. Data augmentation (kun train)
# ===============================
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()  # Ingen augmentation for val/test

# ===============================
# 7. Klar til CNN
# ===============================
# Nu kan du f.eks. bruge:
# model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
#           validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
#           epochs=30)
