#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module principal pour l'entraînement d'Avikam1 LLM
Implémente un Trainer custom avec :
- Gestion optimisée de LoRA
- Support de FlashAttention
- Tracking avec MLflow/W&B
"""

import os
import logging
from typing import Dict, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from src.utils.logger import setup_logger
from src.utils.metrics import compute_metrics
from src.modeling.tokenizer import AvikamTokenizer

logger = setup_logger(__name__)

class AvikamTrainer(Trainer):
    """Trainer personnalisé pour Avikam1 avec optimisations spécifiques"""
    
    def __init__(
        self,
        config: Dict,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        **kwargs
    ):
        # Initialisation des composants
        self.config = config
        self._setup_training()
        self._setup_model()
        self._setup_tokenizer()
        
        # Configuration du data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimisation pour Tensor Cores
        )

        super().__init__(
            model=self.model,
            args=self.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            **kwargs
        )

    def _setup_training(self) -> None:
        """Configure les arguments d'entraînement"""
        self.training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            overwrite_output_dir=True,
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training"].get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation_steps", 1),
            learning_rate=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"].get("weight_decay", 0.01),
            num_train_epochs=self.config["training"]["num_train_epochs"],
            max_steps=self.config["training"].get("max_steps", -1),
            lr_scheduler_type=self.config["training"].get("lr_scheduler_type", "cosine"),
            warmup_ratio=self.config["training"].get("warmup_ratio", 0.05),
            logging_dir=os.path.join(self.config["output_dir"], "logs"),
            logging_strategy="steps",
            logging_steps=self.config["training"].get("logging_steps", 100),
            save_strategy="steps",
            save_steps=self.config["training"].get("save_steps", 500),
            evaluation_strategy="steps" if self.config["data"].get("val_path") else "no",
            eval_steps=self.config["training"].get("eval_steps", 500),
            fp16=self.config["training"].get("fp16", False),
            bf16=self.config["training"].get("bf16", True),
            gradient_checkpointing=self.config["training"].get("gradient_checkpointing", True),
            report_to=["mlflow", "wandb"] if self.config["logging"].get("use_mlflow", False) else [],
            load_best_model_at_end=True if self.config["data"].get("val_path") else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim=self.config["training"].get("optim", "adamw_torch_fused"),
            ddp_find_unused_parameters=False,
            dataloader_num_workers=os.cpu_count(),
        )

    def _setup_model(self) -> None:
        """Charge et configure le modèle"""
        # Configuration de la quantification
        quant_config = None
        if self.config["model"].get("quantize", False):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # Chargement du modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2" if self.config["model"].get("flash_attn", False) else None
        )

        # Application de LoRA si activé
        if self.config["model"].get("use_lora", False):
            lora_config = LoraConfig(
                r=self.config["model"]["lora_rank"],
                lora_alpha=self.config["model"].get("lora_alpha", 32),
                target_modules=self.config["model"].get("lora_target_modules", ["q_proj", "v_proj"]),
                lora_dropout=self.config["model"].get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def _setup_tokenizer(self) -> None:
        """Initialise le tokenizer personnalisé"""
        self.tokenizer = AvikamTokenizer(self.config["model"]["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Surcharge du training_step pour logging custom
        """
        loss = super().training_step(model, inputs)
        
        # Logging des métriques
        if self.state.global_step % self.config["logging"].get("log_steps", 100) == 0:
            self.log({
                "train_loss": loss.item(),
                "learning_rate": self._get_learning_rate()
            })
        
        return loss

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Surcharge de l'évaluation avec des métriques additionnelles
        """
        eval_metrics = super().evaluate(*args, **kwargs)
        
        # Calcul de la perplexité
        perplexity = torch.exp(torch.tensor(eval_metrics["eval_loss"]))
        eval_metrics["perplexity"] = perplexity.item()
        
        return eval_metrics

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Surcharge de la sauvegarde pour gérer correctement LoRA
        """
        if output_dir is None:
            output_dir = self.args.output_dir
            
        if self.config["model"].get("use_lora", False):
            # Sauvegarde uniquement les adaptateurs LoRA
            self.model.save_pretrained(output_dir)
        else:
            # Sauvegarde complète du modèle
            super().save_model(output_dir, _internal_call)
            
        # Sauvegarde toujours le tokenizer
        self.tokenizer.save_pretrained(output_dir)

def find_last_checkpoint(output_dir: str) -> Union[str, None]:
    """Utilitaire pour trouver le dernier checkpoint"""
    return get_last_checkpoint(output_dir)

def train(config_path: str) -> None:
    """Point d'entrée principal pour l'entraînement"""
    try:
        # Chargement de la configuration
        config = load_config(config_path)
        
        # Initialisation du logger
        logger = setup_logger(__name__, config["logging"])
        logger.info("Démarrage de l'entraînement avec la configuration :")
        logger.info(json.dumps(config, indent=2))
        
        # Vérification des checkpoints
        checkpoint = find_last_checkpoint(config["output_dir"])
        if checkpoint:
            logger.info(f"Reprise depuis le checkpoint : {checkpoint}")
        
        # Initialisation du Trainer
        trainer = AvikamTrainer(config)
        
        # Lancement de l'entraînement
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Sauvegarde finale
        trainer.save_model()
        logger.info("Entraînement terminé avec succès !")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration")
    args = parser.parse_args()
    
    train(args.config)
