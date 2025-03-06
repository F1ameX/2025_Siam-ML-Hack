import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from zipfile import ZipFile
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


MODEL_PATH = "models/dae_final.pth"
TRAIN_DIR = "src/train_reduced/"
TEST_DIR = "src/raw_data/test/"
DENOISED_TRAIN_ZIP = "src/raw_data/train_denoised.zip"
DENOISED_TEST_ZIP = "src/raw_data/test_enoised.zip"
SEQUENCE_LENGTH = 100  
CHUNK_SIZE = 500  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.epsilon ** 2))

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
    
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class DAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, bottleneck_dim=32):
        super(DAE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.attn = Attention(hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, hidden_dim)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=5, padding=2)
    
    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        x = F.relu(x).transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.attn(x)
        x = self.bottleneck(x)
        x = self.fc(x)
        x, _ = self.lstm2(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        return x


model = DAE()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def process_file(file_path):
    df = pd.read_csv(file_path, sep="\\s+", names=["time", "pressure"])
    if df.empty or "pressure" not in df or len(df) < SEQUENCE_LENGTH * 5:
        return None
    
    scaler = MinMaxScaler()
    df["pressure"] = scaler.fit_transform(df[["pressure"]])
    sequences = []
    for i in range(len(df) - SEQUENCE_LENGTH):
        seq = df["pressure"].iloc[i : i + SEQUENCE_LENGTH].values
        sequences.append(seq)
    input_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    denoised_chunks = []
    with torch.no_grad():
        for chunk in torch.split(input_tensor, CHUNK_SIZE, dim=0):
            denoised_chunk = model(chunk)
            denoised_chunks.append(denoised_chunk.cpu())
    
    denoised_output = torch.cat(denoised_chunks, dim=0)
    denoised_data = denoised_output.numpy().squeeze()
    
    df = df.iloc[:len(denoised_data)]  
    df["pressure"] = denoised_data

    return df


def process_directory(directory, output_zip):
    with ZipFile(output_zip, "w") as zipf:
        for file_name in tqdm(os.listdir(directory), desc=f"Processing {directory}"):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                df_denoised = process_file(file_path)
                if df_denoised is not None:
                    temp_csv = f"{file_name}.csv"
                    df_denoised.to_csv(temp_csv, sep=" ", index=False, header=False)
                    zipf.write(temp_csv, arcname=temp_csv)
                    os.remove(temp_csv)


process_directory(TRAIN_DIR, DENOISED_TRAIN_ZIP)
process_directory(TEST_DIR, DENOISED_TEST_ZIP)

print("Done")
