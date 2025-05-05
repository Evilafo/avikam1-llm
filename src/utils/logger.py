import logging
from typing import Dict, Any
import mlflow
from datetime import datetime

class TrainingLogger:
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        self.config = config
        self.log_dir = log_dir
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure le système de logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/training_{timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("avikam1")
        
        # Configuration MLflow si activé
        if self.config.get("use_mlflow", False):
            mlflow.set_tracking_uri(self.config["mlflow_uri"])
            mlflow.set_experiment(self.config["experiment_name"])

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log des métriques dans tous les backends"""
        self.logger.info(f"Metrics at step {step}: {metrics}")
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def log_config(self):
        """Log la configuration complète"""
        self.logger.info("Configuration:")
        for k, v in self.config.items():
            self.logger.info(f"{k}: {v}")
        if mlflow.active_run():
            mlflow.log_params(self.config)

    def start_run(self):
        """Démarre une session de logging"""
        if self.config.get("use_mlflow", False):
            mlflow.start_run()
            self.logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")