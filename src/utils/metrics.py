import numpy as np
from sklearn.metrics import accuracy_score

class AvikamMetrics:
    @staticmethod
    def compute_metrics(eval_pred):
        """Calcule les métriques pour l'évaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "perplexity": AvikamMetrics._compute_perplexity(predictions, labels)
        }

    @staticmethod
    def _compute_perplexity(preds, targets, epsilon=1e-10):
        """Calcule la perplexité customisée"""
        cross_entropy = -np.sum(targets * np.log(preds + epsilon), axis=-1)
        return np.exp(np.mean(cross_entropy))