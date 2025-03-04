from clearml import PipelineController

# Create Pipeline
pipe = PipelineController(
    name="Random Forest Training Pipeline",
    project="Random Forest Experiment"
)

# Define execution queue
execution_queue = "rf_queue"

# Add training step
pipe.add_step(
    name="Train Model",
    base_task_project="Random Forest Experiment",
    base_task_name="Random Forest Model Training",
    execution_queue=execution_queue  # 👈 Set execution queue
)

# Add inference step
pipe.add_step(
    name="Run Inference",
    base_task_project="Random Forest Experiment",
    base_task_name="Random Forest Inference",
    execution_queue=execution_queue  # 👈 Set execution queue
)

# Start Pipeline
pipe.start()