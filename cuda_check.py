import os
import glob
import numpy as np
import torch
import librosa
import chromadb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, ASTModel
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

1. SETUP CUDA DEVICE
This ensures your RTX 5070 Ti is doing the heavy lifting
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🚀 GPU Active: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    DEVICE = "cpu"
    print("⚠️ GPU not found. Running on CPU (Slow)")