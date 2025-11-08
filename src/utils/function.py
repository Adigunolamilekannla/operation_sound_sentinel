import pandas as pd 
import numpy as np
from src.utils.exception import CustomException
from src.utils.logger import logging
import os,sys,pickle


def envelope(y,rate,threshold):
    """This function helps strip out very silent audio file"""
    logging.info("Starting audio data envelope")
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of mainUtils class")
        
        # make sure the parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # save the object into the file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of mainUtils class")

    except Exception as e:
        raise CustomException(e, sys)
    