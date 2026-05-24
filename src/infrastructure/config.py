import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_4RAxSFj7CHXN@ep-royal-tree-appd5es0-pooler.c-7.us-east-1.aws.neon.tech/neondb?channel_binding=require&sslmode=require")

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