#!/usr/bin/env python3
# src/inference/api.py

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.logger import get_logger
from mlflow import log_metric, log_params

logger = get_logger(__name__)
app = FastAPI(title="Avikam1 Inference API")

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.on_event("startup")
def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "evilafo/avikam1-7b",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("evilafo/avikam1-7b")
    logger.info("Model loaded successfully")

@app.post("/generate")
async def generate_text(request: InferenceRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Logging to MLflow
    log_metric("generated_tokens", len(outputs[0]))
    log_params(request.dict())
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
