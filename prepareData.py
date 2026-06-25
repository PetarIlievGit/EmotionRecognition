import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/fer2013.csv')
print(df.head())

EMOTIONS = {
    0: 'Лутина',
    1: 'Одвратност',
    2: 'Страв',
    3: 'Среќа',
    4: 'Тага',
    5: 'Изненадување',
    6: 'Неутрална'
}

def pixels_to_image(pixel_string):
    pixels = np.array(pixel_string.split(), dtype='float32')
    return pixels.reshape(48, 48)

train_df = df[df['Usage'] == 'Training']
test_df  = df[df['Usage'] == 'PublicTest']

X_train = np.array([pixels_to_image(p) for p in train_df['pixels']])
y_train = train_df['emotion'].values

X_test  = np.array([pixels_to_image(p) for p in test_df['pixels']])
y_test  = test_df['emotion'].values

X_train = X_train / 255.0
X_test  = X_test  / 255.0
X_train = X_train[..., np.newaxis]  
X_test  = X_test[...,  np.newaxis]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy',  X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy',  y_test)