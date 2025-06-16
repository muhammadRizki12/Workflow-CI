import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import sys
import os

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "obesity_data_preprocessing.csv")

    # Load dataset
    df = pd.read_csv(file_path)

    # Data Preparation
    # Split data menjadi set pelatihan dan set uji
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('ObesityCategory', axis=1), 
        df['ObesityCategory'], 
        test_size=0.3, 
        random_state=42)

    input_example = X_train[0:5]
    
    # Parameter
    penalty = str(sys.argv[1]) if len(sys.argv) > 1 else 'l2'
    solver = str(sys.argv[2]) if len(sys.argv) > 2 else 'lbfgs'
    C = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    max_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    # Modelling
    with mlflow.start_run():

        # Train model
        lg = LogisticRegression(penalty=penalty, solver=solver, C=C, max_iter=max_iter)
        lg.fit(X_train, y_train)

        predicted_obesity = lg.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=lg,
            artifact_path="model",
            input_example=input_example
        )
        lg.fit(X_train, y_train)

        # Log metrics
        accuracy = lg.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)