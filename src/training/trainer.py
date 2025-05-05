import torch
from transformers import TrainingArguments, Trainer
from src.utils.metrics import compute_metrics

class AvikamTrainer:
    def __init__(self, config):
        self.config = config
        self.setup()
        
    def setup(self):
        self.model = self._load_model()
        self.data = self._load_data()
        
    def _load_model(self):
        # Implémentation spécifique
        pass
        
    def train(self):
        args = TrainingArguments(
            output_dir=self.config["output_dir"],
            **self.config["training"]
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            compute_metrics=compute_metrics,
            **self.data
        )
        trainer.train()