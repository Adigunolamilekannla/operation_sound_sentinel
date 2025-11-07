from src.utils.exception import CustomException
from src.utils.logger import logging
from dataclasses import dataclass
import os, sys
from tqdm import tqdm
import librosa
import pandas as pd
from scipy.io import wavfile
from src.utils.dir_manager import MakeDirectory


@dataclass
class DataIngestionConfig:
    train_data_location_bg: str = "data/data/Training data/background"
    train_data_location_gun: str = "data/data/Training data/Gunshot"
    validation_data_location_bg: str = "data/data/Validation data/Background"
    validation_data_location_gun: str = "data/data/Validation data/Gunshot"

    ingested_data_path: str = os.path.join("artifacts", "ingested_data", "data")
    ingested_data_csv_path: str = os.path.join(
        "artifacts", "ingested_data", "csv_file", "audio_df.csv"
    )


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            logging.info("Initializing DataIngestionConfig")
            self.config = config
        except Exception as e:
            raise CustomException(e, sys)

    def process_raw_data(self, csv_path, ingested_data_path):
        """
        This function loops through all audio files, renames them sequentially,
        skips empty files, and saves the valid file paths with labels to a CSV file.
        """
        try:
            logging.info(f"Ingesting data from data sources to {ingested_data_path}")

            # Map folder paths to class labels
            data_loc_dict = {
                self.config.train_data_location_bg: "Background",
                self.config.train_data_location_gun: "Gun_Shot",
                self.config.validation_data_location_bg: "Background",
                self.config.validation_data_location_gun: "Gun_Shot",
            }

            audio_data, audio_label = [], []
            
            

            count = 1
            for filepath, label in data_loc_dict.items():
                if not os.path.exists(filepath):
                    raise ValueError(f"No such directory: {filepath}")

                for filename in tqdm(os.listdir(filepath), desc=f"Processing {label}"):
                    if not filename.lower().endswith(".wav"):
                        continue

                    old_file_path = os.path.join(filepath, filename)
                    new_file_name = f"audio{count}.wav"
                    new_path = os.path.join(filepath, new_file_name)

                    # Safely rename only if not already renamed
                    if not os.path.exists(new_path):
                        os.rename(old_file_path, new_path)

                    # Load and resample audio
                    wav, rate = librosa.load(new_path, sr=48000)

                    # Skip empty audio
                    if wav.size == 0 or os.path.getsize(new_path) == 0:
                        logging.warning(f"Skipping empty file: {new_path}")
                        continue

                    # Save processed audio to artifacts directory
                    os.makedirs(ingested_data_path, exist_ok=True)
                    save_file_path = os.path.join(ingested_data_path, new_file_name)
                    wavfile.write(save_file_path, rate, wav)

                    # Store metadata
                    audio_data.append(save_file_path)
                    audio_label.append(label)
                    count += 1

            # Create CSV
            audio_df = pd.DataFrame({"File_name": audio_data, "Label": audio_label})
            MakeDirectory(csv_path)
            audio_df.to_csv(csv_path, index=False)

            logging.info(f"Data Ingestion Completed. Total files: {len(audio_df)}")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_process_raw_data(self):
        logging.info("Starting Data Ingestion")
        self.process_raw_data(
            self.config.ingested_data_csv_path, self.config.ingested_data_path
        )
