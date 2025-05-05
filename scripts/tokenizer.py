from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Dict, List

class AvikamTokenizer:
    def __init__(self, model_name: str = "evilafo/avikam1-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        """Ajoute des tokens spéciaux pour Avikam1"""
        special_tokens = {
            "additional_special_tokens": ["[USER]", "[BOT]"]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.tokenizer.tokenize(text, **kwargs)
    
    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)