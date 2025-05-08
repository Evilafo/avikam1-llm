import pandas as pd
import yfinance as yf

def load_finance_data(path, tickers, frequency):
    """
    Charge des données financières pour les tickers spécifiés.
    """
    data = []
    for ticker in tickers:
        stock_data = yf.download(ticker, interval=frequency)
        data.append(stock_data.to_string())
    return data

def load_math_data(path):
    """
    Charge des données mathématiques (équations, LaTeX, etc.) depuis un fichier.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

def preprocess_equations(data):
    """
    Prétraite les équations mathématiques pour le modèle.
    """
    processed_data = []
    for line in data:
        if not line.startswith("$"):
            line = f"${line}$"
        processed_data.append(line)
    return processed_data
