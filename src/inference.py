#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'inférence optimisé pour Avikam1 LLM
Fonctionnalités :
- API REST avec FastAPI
- Optimisations GPU (FlashAttention, KV caching)
- Quantification 8/4-bit
- Monitoring temps réel
"""

import os
import time
from typing import Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
import uvicorn
from src.utils.logger import get_logger
from src.utils.metrics import InferenceMetrics

# Configuration des logs
logger = get_logger(__name__)
metrics = InferenceMetrics()

app = FastAPI(
    title="Avikam1 Inference API",
    description="API optimisée pour le modèle Avikam1 LLM",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    """Schéma de requête pour la génération"""
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(50, ge=1, le=1000)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    stream: bool = False

class GenerationResponse(BaseModel):
    """Schéma de réponse standard"""
    generated_text: str
    tokens_generated: int
    inference_time_ms: float
    tokens_per_second: float

@app.on_event("startup")
async def load_model():
    """Charge le modèle et le tokenizer au démarrage"""
    global model, tokenizer
    
    try:
        # Configuration de la quantification
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if os.getenv("QUANTIZE", "true").lower() == "true" else None

        # Chargement du modèle
        model_name = os.getenv("MODEL_NAME", "evilafo/avikam1-7b")
        logger.info(f"Chargement du modèle {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        
        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Modèle chargé avec succès | Mémoire utilisée: %.2fGB", 
                   torch.cuda.max_memory_allocated() / 1e9)

    except Exception as e:
        logger.error("Erreur lors du chargement du modèle: %s", str(e))
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Endpoint principal pour la génération de texte"""
    start_time = time.time()
    
    try:
        # Encodage du prompt
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(model.device)

        # Configuration de la génération
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id
        }

        # Génération synchrone/asynchrone
        if request.stream:
            return streaming_generation(inputs, gen_kwargs)
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )

        # Calcul des métriques
        inference_time = (time.time() - start_time) * 1000
        tokens_per_sec = len(outputs[0]) / (inference_time / 1000)

        # Logging des métriques
        metrics.log(
            prompt_length=len(inputs.input_ids[0]),
            generation_length=len(outputs[0]),
            inference_time=inference_time
        )

        return {
            "generated_text": generated_text,
            "tokens_generated": len(outputs[0]),
            "inference_time_ms": round(inference_time, 2),
            "tokens_per_second": round(tokens_per_sec, 2)
        }

    except torch.cuda.OutOfMemoryError:
        logger.error("Out of memory during generation")
        raise HTTPException(422, "Generation aborted (OOM)")
    except Exception as e:
        logger.error("Generation error: %s", str(e))
        raise HTTPException(500, "Internal server error")

def streaming_generation(inputs: Dict[str, torch.Tensor], gen_kwargs: Dict):
    """Génération en streaming avec yield"""
    streamer = TextIteratorStreamer(tokenizer)
    gen_kwargs["streamer"] = streamer
    
    # Lancement dans un thread séparé
    generation_thread = Thread(target=model.generate, kwargs={
        **{"inputs": inputs.input_ids},
        **gen_kwargs
    })
    generation_thread.start()
    
    # Retour des tokens au fur et à mesure
    def generate():
        try:
            for text in streamer:
                yield f"data: {text}\n\n"
        except Exception as e:
            logger.error("Streaming error: %s", str(e))
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/metrics")
async def get_metrics():
    """Endpoint de monitoring"""
    return {
        "throughput": metrics.throughput(),
        "latency_p95": metrics.latency_p95(),
        "error_rate": metrics.error_rate(),
        "gpu_mem_usage": torch.cuda.memory_allocated() / 1e9
    }

if __name__ == "__main__":
    # Configuration serveur
    server_config = {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 1)),
        "log_level": os.getenv("LOG_LEVEL", "info")
    }
    
    logger.info("Démarrage du serveur sur %s:%d", server_config["host"], server_config["port"])
    uvicorn.run(
        app,
        host=server_config["host"],
        port=server_config["port"],
        workers=server_config["workers"],
        log_level=server_config["log_level"]
    )
