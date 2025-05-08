# Avikam1 LLM - Modèle de Langage pour Evilafo AI
#### Modèle Optimisé pour les Sciences Quantitatives

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/mlflow-%232C3E50.svg?logo=mlflow)](https://mlflow.org/)


**Avikam1-LLM** est un modèle de langage optimisé pour deux principaux cas d’utilisation :
- **Sciences Quantitatives** : la finance quantitative, les mathématiques, l'actuariat, l'informatique, et le trading haute fréquence.
- **Chatbot** : Interactions conversationnelles fluides avec des réponses naturelles et contextuelles tel que [Evilafo AI](https://chat.evilafo.xyz).

Ce projet repose sur l'architecture **LLaMA-2**, avec des améliorations spécifiques pour gérer des tâches avancées tout en restant accessible pour des conversations générales.


## Introduction

Ce projet a pour objectif de fournir un outil puissant et polyvalent à destination des chercheurs et des professionnels évoluant dans des domaines scientifiques exigeants. Avikam1-LLM allie des capacités avancées en sciences quantitatives telles que les mathématiques et la finance à une interface conversationnelle fluide, permettant une interaction naturelle et efficace. Le modèle est capable de traiter des équations complexes, d’analyser des données financières et de répondre avec précision à des questions techniques dans une large variété de domaines.



## ✨ Fonctionnalités Clés

- 🏗️ Fine-tuning efficace avec LoRA/QLoRA, faible consommation de ressources
- ⚡ Inférence rapide via Flash Attention 2
- 📊 Tracking complet avec MLflow
- 🐳 Interface simple pour interagir avec le modèle via HTTP
- 📚 Analyse de séries temporelles et de données boursières.


## 📦 Installation

### Prérequis
- Python 3.10+
- GPU NVIDIA (recommandé) ou CPU
- [PyTorch 2.0+](https://pytorch.org/)

```bash
# Clone du dépôt
git clone https://github.com/evilafo/avikam1-llm
cd avikam1-llm

# Installation des dépendances
pip install -r requirements.txt

# Pour le développement
pip install -r requirements-dev.txt
```

## 🏋️ Entraînement

### Configuration
Éditez le fichier YAML :
```yaml
# configs/default.yaml
model:
  name: "evilafo/avikam1-7b"
  use_lora: true

training:
  learning_rate: 2e-5

quantitative_science:
  enable_math_mode: true
  enable_finance_mode: true
  max_sequence_length: 512
  numerical_precision: "float32"

lora:
  rank: 64
  alpha: 32
lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
  lora_dropout: 0.05

finance_data:
  time_series: true
  frequency: "1min"

metrics:
  perplexity: true
  bleu_score: true
  financial_accuracy: true
```

### Lancement
```bash
python train.py --config configs/lora.yaml
```

## 🚀 Déploiement

### API FastAPI
```bash
python src/inference/api.py
```
**Requête exemple** :
```python
import requests
response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Explique la mécanique quantique", "max_tokens": 150}
)
```

### Avec Docker
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7
COPY . /app
RUN pip install -r /app/requirements.txt
CMD ["python", "/app/src/inference/api.py"]
```

## 📊 Monitoring avec MLflow
Visualisez les métriques :
```bash
mlflow ui --backend-store-uri file:///mlruns
```
![MLflow Dashboard](docs/assets/mlflow-black.svg)



## 🏆 Performances
| Métrique | Valeur |
|----------|--------|
| MMLU (Knowledge) | 65% |
| GSM8K (Math) | 58% |
| Latence (T4 GPU) | 23 tokens/s |

## 🤝 Contribution
1. Forkez le projet
2. Créez une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commitez (`git commit -m 'Add nouvelle fonctionnalite'`)
4. Pushez (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📜 Licence
Apache 2.0 - Voir [LICENSE](LICENSE)

## 📞 Contact
Équipe EvilaFo - [evil2846@gmail.com](mailto:evil2846@gmail.com)

---
