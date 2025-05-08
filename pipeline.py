import os
import subprocess
from train import train_model
from utils.data_loader import load_finance_data, preprocess_equations
from api import app
from fastapi.testclient import TestClient
from utils.math_utils import evaluate_equation
from utils.finance_utils import analyze_stock_data

def run_preprocessing():
    """
    Prétraite les données pour l'entraînement.
    """
    print("Chargement et prétraitement des données...")
    finance_data = load_finance_data(
        path="data/finance",
        tickers=["AAPL", "MSFT"],
        frequency="1min"
    )
    math_data = preprocess_equations(load_math_data(path="data/math"))
    print("Prétraitement terminé.")
    return finance_data, math_data

def run_training():
    """
    Lance l'entraînement du modèle.
    """
    print("Début de l'entraînement du modèle...")
    train_model()
    print("Entraînement terminé.")

def run_evaluation():
    """
    Évalue le modèle entraîné.
    """
    print("Évaluation du modèle...")
    # Évaluation scientifique
    result = evaluate_equation("x**2 + 3*x + 2")
    print(f"Résultat de l'évaluation mathématique : {result}")

    analysis = analyze_stock_data("AAPL", "2023-01-01", "2023-01-10")
    print(f"Analyse financière : {analysis}")

    # Évaluation du chatbot
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "Quelle est la racine carrée de 16 ?"}
    )
    print(f"Réponse du chatbot : {response.json().get('response')}")
    print("Évaluation terminée.")

def deploy_api():
    """
    Déploie l'API REST.
    """
    print("Démarrage de l'API REST...")
    uvicorn_command = ["uvicorn", "api:app", "--reload"]
    subprocess.run(uvicorn_command)
    print("API déployée.")

if __name__ == "__main__":
    # Étape 1 : Prétraitement des données
    run_preprocessing()

    # Étape 2 : Entraînement du modèle
    run_training()

    # Étape 3 : Évaluation du modèle
    run_evaluation()

    # Étape 4 : Déploiement de l'API
    deploy_api()
