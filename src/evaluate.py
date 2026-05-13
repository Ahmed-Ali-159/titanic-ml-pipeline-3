from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import json
import mlflow

# def evaluate_model(model_name: str, trained_pipeline: Pipeline, X_test, y_test):
def evaluate_model(model_name, trained_pipeline, X_test, y_test, run_id=None):
    
    preds = trained_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # # Log metrics to MLflow
    # mlflow.log_metric("accuracy", acc)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    

    # Save metrics to file for DVC to track
    metrics = {"accuracy": round(acc, 3)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("[✓] Metrics saved → metrics.json")

