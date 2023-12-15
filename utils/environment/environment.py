import os 
import random 
import torch 
import numpy as np

def setup_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

def create_checkpoint_folder(checkpoint_folder):
    # Create the checkpoint folder if it doesn't exist
    os.makedirs(checkpoint_folder, exist_ok=True)
