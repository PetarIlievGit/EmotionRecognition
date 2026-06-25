import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

EMOTIONS = ['Лутина','Одвр.','Страв','Среќа','Тага','Изн.','Неутр.']

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
y_test_cat = to_categorical(y_test, num_classes=7)  

model = load_model('models/emotion_model.h5')


loss, acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {acc*100:.2f}%")

y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=EMOTIONS))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap='Blues')
plt.title('Confusion Matrix — Емоции')
plt.ylabel('Вистинска'); plt.xlabel('Предвидена')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()