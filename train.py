from src.training.trainer import Trainer
from configs import load_config

def main():
    config = load_config("configs/default.yaml")
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()