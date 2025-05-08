from src.training.trainer import Trainer
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from utils.data_loader import load_finance_data, load_math_data, preprocess_equations
from utils.math_utils import evaluate_equation
from utils.finance_utils import analyze_stock_data

# Charger la configuration
with open("default.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_training_data():
    """
    Charge les données d'entraînement en fonction de la configuration.
    """
    if config["quantitative_science"]["enable_finance_mode"]:
        # Charger des données financières
        data = load_finance_data(
            path=config["data_path"],
            tickers=config["finance_data"]["tickers"],
            frequency=config["finance_data"]["frequency"]
        )
        print("Données financières chargées.")
    elif config["quantitative_science"]["enable_math_mode"]:
        # Charger des données mathématiques (équations, LaTeX, etc.)
        data = load_math_data(path=config["data_path"])
        data = preprocess_equations(data)  # Prétraitement pour LaTeX et équations
        print("Données mathématiques chargées et prétraitées.")
    else:
        # Charger des données génériques ou des textes non spécialisés
        data = load_generic_data(path=config["data_path"])
        print("Données génériques chargées.")
    return data

def load_generic_data(path):
    """
    Charge des données génériques depuis un fichier texte.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

def train_model():
    """
    Entraîne le modèle en utilisant les paramètres spécifiés dans la configuration.
    """
    # Initialiser le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Charger les données d'entraînement
    data = load_training_data()

    # Configuration LoRA/QLoRA
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Définir l'optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Boucle d'entraînement
    for epoch in range(config["max_epochs"]):
        print(f"Début de l'époque {epoch + 1}/{config['max_epochs']}")
        for batch in data:
            # Tokenisation des données
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=config["quantitative_science"]["max_sequence_length"])
            
            # Calcul de la perte et mise à jour des poids
            outputs = model(**inputs,
