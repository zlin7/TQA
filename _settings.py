import os
import getpass
import sys


_ON_SERVER = True

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

#==============================Data Related
COVID_NAME = 'COVID'
MIMIC_NAME = 'MIMIC'
EEG_NAME = 'EEG'
GEFCom_NAME = 'GEFCom'


WORKSPACE = os.path.join(__CUR_FILE_PATH, "Temp")
DATA_PATH = os.path.join(WORKSPACE, 'data')
_PERSIST_PATH = os.path.join(WORKSPACE, 'cache')
LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
BASE_MODEL_PATH = os.path.join(WORKSPACE, 'base_models')
if not os.path.isdir(BASE_MODEL_PATH): os.makedirs(BASE_MODEL_PATH)
RANDOM_SEED = 7

NCOLS = 80


METHOD_PLACEHOLDER = 'TQA'
