#!/usr/bin/env python3
"""
Convertit les checkpoints PyTorch vers le format HuggingFace
Usage:
  python convert_hf.py --input_dir ./ckpt --output_dir ./hf_model
"""
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def convert_to_hf(input_dir: str, output_dir: str, model_name: str = None):
    """Convertit un checkpoint custom en modèle HF"""
    model_path = input_dir if model_name is None else model_name
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configuration spéciale pour Avikam1
    model.config.update({"model_type": "avikam1-llm"})
    
    # Sauvegarde
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modèle converti avec succès dans {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    
    convert_to_hf(args.input_dir, args.output_dir, args.model_name)