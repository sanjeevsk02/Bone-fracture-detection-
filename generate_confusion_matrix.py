import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
# Define load_path function (copied from training_part.py)
def load_path(path):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                path_p = folder + '/' + str(body)
                for id_p in os.listdir(path_p):
                    patient_id = id_p
                    path_id = path_p + '/' + str(id_p)
                    for lab in os.listdir(path_id):
                        if lab.split('_')[-1] == 'positive':
                            label = 'fractured'
                        elif lab.split('_')[-1] == 'negative':
                            label = 'normal'
                        path_l = path_id + '/' + str(lab)
                        for img in os.listdir(path_l):
                            img_path = path_l + '/' + str(img)
                            dataset.append(
                                {
                                    'label': body,
                                    'image_path': img_path
                                }
                            )
    return dataset

# Load the saved model
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(THIS_FOLDER, 'weights/ResNet50_BodyParts.h5')
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Load dataset and prepare test dataframe
image_dir = os.path.join(THIS_FOLDER, 'Dataset')  # Adjust if your dataset path differs
data = load_path(image_dir)

labels = []
filepaths = []
for row in data:
    labels.append(row['label'])
    filepaths.append(row['image_path'])

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
images = pd.concat([filepaths, labels], axis=1)

# Split to get test_df (same as in training_part.py)
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# Test generator (same as in training_part.py)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Get predictions
y_pred = np.argmax(model.predict(test_images), axis=1)
y_true = test_images.classes


# Compute metrics (average='macro' for multi-class balance)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
# Confusion Matrix
labels = ["Elbow", "Hand", "Shoulder"]  # From categories_parts in predictions.py
cm = confusion_matrix(y_true, y_pred)

# Create directory if it doesn't exist
plots_dir = os.path.join(THIS_FOLDER, 'plots/BodyParts/')
os.makedirs(plots_dir, exist_ok=True)

# Plot and Save
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for Body Part Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
save_path = os.path.join(plots_dir, 'confusion_matrix.jpeg')
plt.savefig(save_path)
plt.show()
print(f"Confusion matrix saved to {save_path}")