import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')

DATABASE_URL = os.getenv("DATABASE_URL")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'reports'), exist_ok=True)

SEQUENCE_LENGTH = 10
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)