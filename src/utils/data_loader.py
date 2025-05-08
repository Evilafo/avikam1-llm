import pandas as pd
import yfinance as yf

def load_finance_data(path, tickers, frequency):
    """
    Charge des données financières pour les tickers spécifiés.
    Args:
        path (str): Chemin vers le répertoire des données (non utilisé ici, mais peut être utile pour d'autres formats).
        tickers (list): Liste des symboles boursiers (e.g., ["AAPL", "MSFT"]).
        frequency (str): Fréquence des données (e.g., "1min", "1d").
    Returns:
        list: Données financières sous forme de chaînes de caractères.
    """
    data = []
    for ticker in tickers:
        stock_data = yf.download(ticker, interval=frequency)
        data.append(stock_data.to_string())
    return data

def load_math_data(path):
    """
    Charge des données mathématiques (équations, LaTeX, etc.) depuis un fichier.
    Args:
        path (str): Chemin vers le fichier contenant les données mathématiques.
    Returns:
        list: Lignes du fichier sous forme de liste.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

def preprocess_equations(data):
    """
    Prétraite les équations mathématiques pour le modèle.
    Args:
        data (list): Liste de chaînes de caractères représentant des équations.
    Returns:
        list: Équations prétraitées avec des balises LaTeX si nécessaire.
    """
    processed_data = []
    for line in data:
        if not line.startswith("$"):
            line = f"${line}$"
        processed_data.append(line.strip())
    return processed_data
