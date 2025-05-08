import pandas as pd
import yfinance as yf

def analyze_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Analyse les données financières d'une action.
    Args:
        ticker (str): Symbole boursier (e.g., "AAPL").
        start_date (str): Date de début au format "YYYY-MM-DD".
        end_date (str): Date de fin au format "YYYY-MM-DD".
    Returns:
        dict: Analyse des données financières.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        analysis = {
            "mean_price": data["Close"].mean(),
            "volatility": data["Close"].std(),
            "total_volume": data["Volume"].sum()
        }
        return analysis
    except Exception as e:
        raise ValueError(f"Erreur lors de l'analyse des données financières : {e}")
