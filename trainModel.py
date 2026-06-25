import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test  = np.load('data/X_test.npy')
y_test  = np.load('data/y_test.npy')

NUM_CLASSES = 7
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

datagen = ImageDataGenerator(
    rotation_range     = 10,
    horizontal_flip    = True,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    zoom_range         = 0.1,
    fill_mode          = 'nearest'
)
datagen.fit(X_train)

def build_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_model()
model.summary()

model.compile(
    optimizer = Adam(learning_rate=0.0005),  
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

callbacks = [
    ModelCheckpoint(
        'models/emotion_model.h5',
        save_best_only = True,
        monitor        = 'val_accuracy',
        verbose        = 1
    ),
    EarlyStopping(
        patience             = 15,   
        restore_best_weights = True,
        monitor              = 'val_accuracy',
        verbose              = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 5,
        min_lr   = 1e-7,
        verbose  = 1
    )
]

print("\n Training \n")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=64),
    epochs           = 80,
    validation_data  = (X_test, y_test_cat),
    callbacks        = callbacks,
    class_weight     = class_weight_dict   
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy низ епохи')
ax1.set_xlabel('Епоха'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss низ епохи')
ax2.set_xlabel('Епоха'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=120)
plt.show()
print("Finished")