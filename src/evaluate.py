from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json

def evaluate_model(model_name: str, trained_pipeline: Pipeline, X_test, y_test):
    
    preds = trained_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Save metrics to file for DVC to track
    metrics = {"accuracy": round(acc, 3)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("[✓] Metrics saved → metrics.json")

