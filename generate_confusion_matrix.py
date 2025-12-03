import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === PATHS ===
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(THIS_FOLDER, "Dataset", "train_valid")  # your real folder
WEIGHTS   = os.path.join(THIS_FOLDER, "weights")
PLOTS     = os.path.join(THIS_FOLDER, "plots")
os.makedirs(PLOTS, exist_ok=True)

#Load all images 
def load_images():
    data = []
    for body in ["Elbow", "Hand", "Shoulder"]:
        body_path = os.path.join(DATA_ROOT, body)
        if not os.path.exists(body_path): continue
        for patient in os.listdir(body_path):
            patient_path = os.path.join(body_path, patient)
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                label = 'fractured' if 'positive' in study else 'normal'
                for img in os.listdir(study_path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data.append({
                            'path': os.path.join(study_path, img),
                            'body': body,
                            'fracture': label
                        })
    return pd.DataFrame(data)

df = load_images()


_, test_df = train_test_split(df, train_size=0.9, shuffle=True, random_state=1)

gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# --------------------------------------------------
# 1. BODY PART CLASSIFICATION
# --------------------------------------------------
print("\n" + "="*70)
print("     BODY PART CLASSIFICATION (Stage 1)")
print("="*70)

model = tf.keras.models.load_model(os.path.join(WEIGHTS, "ResNet50_BodyParts.h5"))

flow = gen.flow_from_dataframe(
    test_df, x_col='path', y_col='body',
    target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False
)

pred = np.argmax(model.predict(flow), axis=1)
true = flow.classes
labels = ["Elbow", "Hand", "Shoulder"]

acc = np.mean(pred == true) * 100
print(f"Test Accuracy: {acc:.2f}%")
print(classification_report(true, pred, target_names=labels, digits=4))

os.makedirs(os.path.join(PLOTS, "BodyParts"), exist_ok=True)
cm = confusion_matrix(true, pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Body Part Classification")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "BodyParts", "confusion_matrix.jpeg"), dpi=300)
plt.show()

# --------------------------------------------------
# 2. FRACTURE DETECTION (Elbow / Hand / Shoulder)
# --------------------------------------------------
for part in ["Elbow", "Hand", "Shoulder"]:
    print("\n" + "="*70)
    print(f"     FRACTURE DETECTION - {part.upper()}")
    print("="*70)

    part_df = test_df[test_df['body'] == part].copy()

    model = tf.keras.models.load_model(os.path.join(WEIGHTS, f"ResNet50_{part}_frac.h5"))

    flow = gen.flow_from_dataframe(
        part_df, x_col='path', y_col='fracture',
        target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False
    )

    pred = np.argmax(model.predict(flow), axis=1)
    true = flow.classes
    labels = ["Normal", "Fractured"]

    acc = np.mean(pred == true) * 100
    print(f"{part} Test Accuracy: {acc:.2f}%")
    print(classification_report(true, pred, target_names=labels, digits=4))

    os.makedirs(os.path.join(PLOTS, f"Fracture_{part}"), exist_ok=True)
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {part} Fracture Detection")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"Fracture_{part}", "confusion_matrix.jpeg"), dpi=300)
    plt.show()

print("\nAll real results + plots saved in 'plots' folder!")