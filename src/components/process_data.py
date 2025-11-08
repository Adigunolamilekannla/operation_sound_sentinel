from src.utils.exception import CustomException
from src.utils.logger import logging
from dataclasses import dataclass
import os, sys, gc
from tqdm import tqdm
import librosa
import pandas as pd
from scipy.io import wavfile
from src.utils.function import envelope, save_object
from sklearn.model_selection import train_test_split
import numpy as np
import psutil
from python_speech_features import mfcc


@dataclass
class ProcessDataConfig:
    process_data_path: str = os.path.join("artifacts", "ingested_data", "process_data","raw_audio")
    train_process_data_path: str = os.path.join(
        "artifacts", "ingested_data", "process_data", "model_ready_data" ,"train_audio_df.pkl"
    )
    test_process_data_path: str = os.path.join(
        "artifacts", "ingested_data", "process_data",  "model_ready_data" ,"test_audio_df.pkl"
    )
    ingested_data_path: str = os.path.join("artifacts", "ingested_data", "data")
    ingested_data_csv_path: str = os.path.join(
        "artifacts", "ingested_data", "csv_file", "audio_df.csv"
    )


class AudioConfig:
    def __init__(self, mode="conv", nfilt=40, nfeat=13, nfft=2048, rate=48000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate * 4)  # 4 seconds per sample


class ProcessData:
    def __init__(self, config: ProcessDataConfig):
        try:
            logging.info("Initializing ProcessDataConfig")
            self.config = config
            self.audio_config = AudioConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def process_data(self, csv_path, process_path):
        """Load ingested data, clean it, and save to processed path."""
        try:
            os.makedirs(process_path, exist_ok=True)
            audio_df = pd.read_csv(csv_path)

            if len(os.listdir(process_path)) == 0:
                for f in tqdm(audio_df["File_name"]):
                    f = f.strip()

                    if not os.path.exists(f) or os.path.getsize(f) == 0:
                        continue

                    
                    signal, rate = librosa.load(f, sr=self.audio_config.rate)

                    mask = envelope(signal, rate, 0.0005)
                    clean_signal = signal[mask]

                    if len(clean_signal) == 0:
                        continue

                    clean_path = os.path.join(process_path, os.path.basename(f))
                    wavfile.write(clean_path, rate, clean_signal.astype(np.float32))

            audio_df.set_index("File_name", inplace=True)
            audio_df["Label"] = audio_df["Label"].map({"Background": 0, "Gun_Shot": 1})

            train_df, test_df = train_test_split(audio_df, test_size=0.2, random_state=42)
            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)

    def build_all_feat(self, audio_df_, process_path):
        try:
            gc.collect()
            print("Available memory:", psutil.virtual_memory().available / 1e9, "GB")

            X, y = [], []
            _min, _max = float("inf"), -float("inf")

            for file in tqdm(audio_df_.index):
                filename = os.path.basename(file)
                filepath = os.path.join(process_path, filename)

                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    continue

                wav, rate = librosa.load(filepath, sr=self.audio_config.rate) 
                if len(wav) < self.audio_config.step:
                    continue

                wav = wav[:self.audio_config.step]

                X_sample = mfcc(
                    wav, rate,
                    numcep=self.audio_config.nfeat,
                    nfilt=self.audio_config.nfilt,
                    nfft=self.audio_config.nfft
                ).T

                _min = min(np.amin(X_sample), _min)
                _max = max(np.amax(X_sample), _max)

                X.append(X_sample)
                y.append(audio_df_.at[file, "Label"])

            
            X = np.array([x for x in X if x.shape == X[0].shape])
            y = np.array(y[:len(X)])

            X = (X - _min) / (_max - _min)
            X = X[..., np.newaxis]

            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_process_data(self):
        try:
            logging.info("Starting Data Processing")

            train_df, test_df = self.process_data(
                self.config.ingested_data_csv_path, self.config.process_data_path
            )
            if not os.path.exists(self.config.train_process_data_path):
                X_train, y_train = self.build_all_feat(train_df, self.config.process_data_path)
                X_test, y_test = self.build_all_feat(test_df, self.config.process_data_path)
                save_object(self.config.train_process_data_path, (X_train, y_train))
                save_object(self.config.test_process_data_path, (X_test, y_test))

           
            

            logging.info("Data processing completed successfully")

        except Exception as e:
            raise CustomException(e, sys)
