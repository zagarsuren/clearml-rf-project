import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from clearml import Task, Logger
from utils.data_loader import load_data
from utils.model_utils import save_model

# Initialize ClearML Task
task = Task.init(
    project_name="Random Forest Experiment",
    task_name="Random Forest Model Training",
    task_type=Task.TaskTypes.training
)
logger = task.get_logger()

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Set Hyperparameters
params = {
    "n_estimators": 100,
    "max_depth": 4,
    "random_state": 42
}
task.connect(params)

# Train Model
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.report_scalar("Accuracy", "Test", iteration=1, value=accuracy)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save Model
save_model(model, "models/random_forest.pkl")
task.upload_artifact(name="RandomForest Model", artifact_object="models/random_forest.pkl")

# --- ADD TRAINING PLOTS ---

# 1️⃣ Feature Importance Plot
feature_importance = model.feature_importances_
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.savefig("models/feature_importance.png")

# Log the plot in ClearML
# Load and convert image to PIL format before logging
feature_img = Image.open("models/feature_importance.png")
logger.report_image("Feature Importance", "Feature Importance Plot", iteration=1, image=feature_img)

print("Plots successfully logged in ClearML.")