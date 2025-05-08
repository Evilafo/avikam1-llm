import pytest
from utils.math_utils import evaluate_equation
from utils.finance_utils import analyze_stock_data
from api import app
from fastapi.testclient import TestClient

# Client pour tester l'API
client = TestClient(app)

# Test pour évaluer une équation mathématique
def test_evaluate_equation():
    equation = "x**2 + 3*x + 2"
    result = evaluate_equation(equation)
    assert isinstance(result, float), "Le résultat doit être un nombre flottant."
    assert result == 6.0, "L'évaluation de l'équation est incorrecte."

# Test pour analyser des données financières
def test_analyze_stock_data():
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    analysis = analyze_stock_data(ticker, start_date, end_date)
    assert "mean_price" in analysis, "La clé 'mean_price' est manquante dans l'analyse."
    assert "volatility" in analysis, "La clé 'volatility' est manquante dans l'analyse."
    assert "total_volume" in analysis, "La clé 'total_volume' est manquante dans l'analyse."

# Test pour l'API : Évaluer une équation
def test_api_evaluate_equation():
    response = client.post(
        "/evaluate-equation",
        headers={"X-API-Key": "your_secret_api_key"},
        json={"equation": "x**2 + 3*x + 2"}
    )
    assert response.status_code == 200, "La requête a échoué."
    assert "result" in response.json(), "La réponse ne contient pas la clé 'result'."
    assert response.json()["result"] == 6.0, "Le résultat de l'évaluation est incorrect."

# Test pour l'API : Analyser une action boursière
def test_api_analyze_stock():
    response = client.post(
        "/analyze-stock",
        headers={"X-API-Key": "your_secret_api_key"},
        json={"ticker": "AAPL", "start_date": "2023-01-01", "end_date": "2023-01-10"}
    )
    assert response.status_code == 200, "La requête a échoué."
    assert "analysis" in response.json(), "La réponse ne contient pas la clé 'analysis'."
    analysis = response.json()["analysis"]
    assert "mean_price" in analysis, "La clé 'mean_price' est manquante dans l'analyse."
# Test pour l'API : Chatbot
def test_api_chatbot():
    response = client.post(
        "/chat",
        json={"message": "Quelle est la racine carrée de 16 ?"}
    )
    assert response.status_code == 200, "La requête a échoué."
    assert "response" in response.json(), "La réponse ne contient pas la clé 'response'."
    assert response.json()[
