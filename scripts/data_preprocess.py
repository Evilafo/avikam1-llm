from datasets import load_dataset
from src.modeling.tokenizer import Tokenizer

def preprocess(data_path, output_dir):
    tokenizer = Tokenizer.from_pretrained("evilafo/avikam1-7b")
    dataset = load_dataset("json", data_files=data_path)
    
    # Tokenisation et sauvegarde
    dataset.map(lambda x: tokenizer(x["text"]))
    dataset.save_to_disk(output_dir)