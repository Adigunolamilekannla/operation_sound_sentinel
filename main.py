from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.components.process_data import ProcessData,ProcessDataConfig


data_config = DataIngestionConfig()
ingest_data = DataIngestion(config=data_config)
ingest_data.initiate_ingest_raw_data()

process_config = ProcessDataConfig()
process_data = ProcessData(config=process_config)
process_data.initiate_process_data()