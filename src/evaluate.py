from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# def evaluate_models(trained_models: dict, X_test, y_test) -> None:
#     """Print accuracy, classification report, and confusion matrix."""
#     for name, pipeline in trained_models.items():
#         preds = pipeline.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         print(f"\n=== {name} Evaluation ===")
#         print(f"Accuracy: {acc:.4f}")
#         print(classification_report(y_test, preds))
#         print("Confusion Matrix:")
#         print(confusion_matrix(y_test, preds))

def evaluate_model(model_name: str, trained_pipeline: Pipeline, X_test, y_test):
    
    preds = trained_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

