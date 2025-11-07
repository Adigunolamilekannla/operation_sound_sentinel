from src.components.data_ingestion import DataIngestion,DataIngestionConfig


data_config = DataIngestionConfig()
ingest_data = DataIngestion(config=data_config)
ingest_data.initiate_process_raw_data()