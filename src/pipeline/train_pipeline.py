from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.process_data import ProcessData, ProcessDataConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys


class TrainPipeline:
    def __init__(self):
        """Initialize all configuration classes"""
        self.data_config = DataIngestionConfig()
        self.process_config = ProcessDataConfig()
        self.model_config = ModelTrainerConfig()

    def start_training_model(self):
        """
        Runs the full pipeline:
        1. Data ingestion
        2. Data processing
        3. Model training and evaluation
        """
        logging.info(" Starting full training pipeline...")
        try:
            # STEP 1: Data Ingestion
            logging.info(" Step 1: Data Ingestion started...")
            data_ingestor = DataIngestion(config=self.data_config)
            data_ingestor.initiate_ingest_raw_data()
            logging.info(" Data ingestion completed successfully.")

            # STEP 2: Data Processing
            logging.info(" Step 2: Data Processing started...")
            processor = ProcessData(config=self.process_config)
            processor.initiate_process_data()
            logging.info(" Data processing completed successfully.")

            # STEP 3: Model Training
            logging.info(" Step 3: Model Training started...")
            model_trainer = ModelTrainer(config=self.model_config)
            model_trainer.initiate_train_and_evaluate_model()
            logging.info(" Model training completed successfully.")

            logging.info(" All pipeline steps completed successfully!")
            return True

        except Exception as e:
            logging.error(" Error occurred during training pipeline execution.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Optional: Run the pipeline directly from this file
    try:
        pipeline = TrainPipeline()
        pipeline.start_training_model()
    except Exception as e:
        print(f"Pipeline failed: {e}")
