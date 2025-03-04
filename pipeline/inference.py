import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from clearml import Task
from utils.model_utils import load_model

# Initialize ClearML Task
task = Task.init(
    project_name="Random Forest Experiment",
    task_name="Random Forest Inference",
    task_type=Task.TaskTypes.inference
)

# Load trained model
model = load_model("models/random_forest.pkl")

# Sample input
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample_input)

print(f"Predicted class: {prediction}")
