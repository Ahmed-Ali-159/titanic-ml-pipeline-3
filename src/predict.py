import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Load model from MLflow model registry using alias
model = mlflow.sklearn.load_model("models:/titanic-classifier@production")

# Sample input
sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}])

prediction = model.predict(sample)
print(f"Survived: {prediction[0]}")