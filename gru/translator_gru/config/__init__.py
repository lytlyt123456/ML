import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS = 0
EOS = 1
MAX_SEQ_LENGTH = 10
DATASET_PATH = 'dataset/eng-cmn.txt'
OUTPUT_DIR = 'output'
EMBED_DIM = 256
LEARNING_RATE = 1e-2
NUM_EPOCHS = 300