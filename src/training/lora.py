from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

def setup_lora(model: PreTrainedModel, config: dict) -> PreTrainedModel:
    """Configure LoRA pour le modèle"""
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]  # Couches à entraîner complètement
    )
    return get_peft_model(model, lora_config)

def print_trainable_params(model: PreTrainedModel) -> None:
    """Affiche le nombre de paramètres entraînables"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables: {trainable:,} ({100*trainable/total:.2f}% du total)")