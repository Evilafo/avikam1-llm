from transformers import AutoModelForCausalLM

class AvikamModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def forward(self, inputs):
        return self.model(**inputs)