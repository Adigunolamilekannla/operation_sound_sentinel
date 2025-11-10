import os
import sys
import torch
import random
import numpy as np
import mlflow
from dotenv import load_dotenv
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.model import classification_report, train_model, predict_audio
from src.utils.function import load_object
from src.utils.dir_manager import MakeDirectory


# ===============================
#   Load environment variables
# ===============================
load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ===============================
#   Configuration
# ===============================
@dataclass
class ModelTrainerConfig:
    train_process_data_path: str = os.path.join(
        "artifacts", "ingested_data", "process_data", "model_ready_data", "train_audio_df.pkl"
    )
    test_process_data_path: str = os.path.join(
        "artifacts", "ingested_data", "process_data", "model_ready_data", "test_audio_df.pkl"
    )
    trained_model_path_artifact: str = os.path.join("artifacts", "trained_model", "model.pt")
    trained_model_path: str = os.path.join("final_model", "model.pt")


# ===============================
#   Model Trainer Class
# ===============================
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        """Initialize the ModelTrainer class."""
        try:
            logging.info("Initializing ModelTrainerConfig")
            self.config = config

            # Reproducibility setup
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    #   MLflow Tracking
    # -----------------------------
    def mlflow_tracking(self, model_name, model, train_metrics, test_metrics, register_model=True):
        """Logs model, parameters, and metrics to MLflow."""
        try:
            logging.info(f"Starting MLflow tracking for {model_name}")

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment("Operation_Sound_Sentinel")

            with mlflow.start_run(run_name=model_name):
                # Log PyTorch model
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=model_name if register_model else None,
                )

                # Log training and test metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)

            logging.info(f"MLflow tracking completed for {model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    #   Training and Evaluation
    # -----------------------------
    def train_and_evaluate_model(self, train_data_path, test_data_path):
        """Trains, evaluates, and logs model results."""
        try:
            logging.info("Model training started")

            # Load preprocessed data
            X_train, y_train = load_object(train_data_path)
            X_test, y_test = load_object(test_data_path)

            # Convert to torch tensors
            X_train = torch.tensor(X_train).float()
            y_train = torch.tensor(y_train).long()
            X_test = torch.tensor(X_test).float()
            y_test = torch.tensor(y_test).long()

            # Create DataLoaders
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False, drop_last=True)

            # Select device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Train model
            net = train_model(train_loader, device)

            # Predictions
            y_train_pred, y_train_true = predict_audio(net=net, device=device, data_loader=train_loader)
            y_test_pred, y_test_true = predict_audio(net=net, device=device, data_loader=test_loader)

            # Compute metrics
            y_train_pred_metric = classification_report(y_train_true, y_train_pred)
            y_test_pred_metric = classification_report(y_test_true, y_test_pred)

            # MLflow tracking
            self.mlflow_tracking(
                "Operation_Sound_Sentinel_Model",
                net,
                y_train_pred_metric,
                y_test_pred_metric,
                register_model=True
            )

            # Save model
            MakeDirectory(self.config.trained_model_path_artifact)
            MakeDirectory(self.config.trained_model_path)

            torch.save(net.state_dict(), self.config.trained_model_path_artifact)
            torch.save(net.state_dict(), self.config.trained_model_path)

            logging.info(
                f"Model saved at:\n  - {self.config.trained_model_path_artifact}\n  - {self.config.trained_model_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    #   Pipeline Entry Point
    # -----------------------------
    def initiate_train_and_evaluate_model(self):
        """Runs the full training and evaluation pipeline."""
        self.train_and_evaluate_model(
            self.config.train_process_data_path,
            self.config.test_process_data_path
        )




        
        