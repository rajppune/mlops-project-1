import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Dummy model
model = RandomForestClassifier()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo-model-registry")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.register_model("runs:/{run_id}/model", "DemoModel")
