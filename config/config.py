import pathlib

#import finrl

import pandas as pd
import datetime
import time
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10



PATH = os.getcwd()
TRAINED_MODEL_DIR = os.path.join(PATH,"trained_models")
print("PATH                 ",PATH)
print("TRAINED_MODEL_DIR    ",TRAINED_MODEL_DIR)

if not os.path.exists(TRAINED_MODEL_DIR):
    print("Creat                  ", TRAINED_MODEL_DIR)
    os.mkdir(TRAINED_MODEL_DIR)


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent
#TRAINED_MODEL_DIR = PATH / "trained_models"

# data
DATASET_DIR = os.path.join(PATH,"data")
print("DATASET_DIR          ",DATASET_DIR)
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

now = datetime.datetime.now()
today = datetime.datetime.now()
date_time = today.strftime("%Y-%m-%d#%H%M%S")
print("date and time:       ",date_time)

TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, date_time)

#TRAINED_MODEL_DIR = f"trained_models/{now}"
#TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, now.ctime())
#TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, "2020-11-29T19:10:29.054778")
#TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, "2020-11-29T19-10-29")
#TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, "tototest")
#print("minute    ",now.minute.isoformat())
#print("hour      ",now.hour)

print("TRAINED_MODEL_DIR    ",TRAINED_MODEL_DIR)
print("TRAINING_DATA_FILE   ",TRAINING_DATA_FILE)


os.makedirs(TRAINED_MODEL_DIR)



TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
TESTING_DATA_FILE = "test.csv"



    
    
    
 