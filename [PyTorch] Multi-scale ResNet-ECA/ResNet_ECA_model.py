# ## Model training and test
# #### FIF+MS-ResNet+ECA (Our proposed model)
# #### The data are after the normalization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import warnings
from scipy.io import loadmat
import os
import pickle
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## 1. Data loading

snr_levels = ["50dB", "40dB", "30dB", "20dB"]
sig_no = 800 # Select the first 800 rows, the rest 200 rows are for test datset
num_channels = 10  # Number of IMF channels extracted via FIF; use 1 when feeding raw PQD signals
num_classes = 6;
sequence_length = 640
PQD_labels = {
    'Normal': [0, 0, 0, 0, 0, 0],
    'Flicker': [1, 0, 0, 0, 0, 0],
    'Swell': [0, 1, 0, 0, 0, 0],
    'Sag': [0, 0, 1, 0, 0, 0],
    'Interruption': [0, 0, 0, 1, 0, 0],
    'Harmonics': [0, 0, 0, 0, 1, 0],
    'Oscillatory transient': [0, 0, 0, 0, 0, 1],
    'Flicker+Swell': [1, 1, 0, 0, 0, 0],
    'Flicker+Sag': [1, 0, 1, 0, 0, 0],
    'Flicker+Harmonics': [1, 0, 0, 0, 1, 0],
    'Flicker+Transient': [1, 0, 0, 0, 0, 1],
    'Swell+Harmonics': [0, 1, 0, 0, 1, 0],
    'Swell+Transient': [0, 1, 0, 0, 0, 1],
    'Sag+Harmonics': [0, 0, 1, 0, 1, 0],
    'Sag+Transient': [0, 0, 1, 0, 0, 1],
    'Interruption+Harmonics': [0, 0, 0, 1, 1, 0],
    'Flicker+Swell+Harmonics': [1, 1, 0, 0, 1, 0],
    'Flicker+Sag+Harmonics': [1, 0, 1, 0, 1, 0],
    'Swell+Harmonics+Transient': [0, 1, 0, 0, 1, 1],
    'Sag+Harmonics+Transient': [0, 0, 1, 0, 1, 1]
}

class PQDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    X = []
    y = []
    for snr in snr_levels:
        folder_path = os.path.join(data_path, snr)
        for file in os.listdir(folder_path):
            if file.endswith('.xlsx'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_excel(file_path, header=None, engine='openpyxl')

                for i in range(sig_no):
                    emd_signal = df.iloc[i*num_channels:(i+1)*num_channels, :].values  # 10 channels
                    X.append(emd_signal)

                label_name = os.path.splitext(file)[0]
                y.append(PQD_labels[label_name])
    
    X = np.array(X)  #
    y = np.repeat(y, X.shape[0] // len(y), axis=0)  # Repeat labels accordingly

    # Reshape for PyTorch input (batch_size, channels, sequence_length)
    X = X.reshape(-1, num_channels, sequence_length)
    return X, y

# Load the FIF dataset (uploaded to Google Drive)
with open('FIF_train_1.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('FIF_val_1.pkl', 'rb') as f:
    val_dataset = pickle.load(f) 

# Extract X (signals) and y(labels) from datasets
X_train, y_train = train_dataset.X, train_dataset.y
X_val, y_val = val_dataset.X, val_dataset.y

# Move datasets to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Update datasets with GPU tensors
train_dataset.X, train_dataset.y = X_train, y_train
val_dataset.X, val_dataset.y = X_val, y_val

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Create DataLoaders
batch_size = 120
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ## 2. Model

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

# ECA module implementation
class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        # Adaptive kernel size based on channel dimension
        kernel_size = int(abs(math.log(channels, 2) + b) / gamma)
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [batch, channels, seq_len]
        batch, channels, _ = x.size()
        y = self.avg_pool(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        y = self.sigmoid(y)
        return x * y

class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:,:,0:-d] + out if d > 0 else residual + out
        out1 = self.relu(out1)

        return out1

class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out if d > 0 else residual + out
        out1 = self.relu(out1)

        return out1

class MSResNet_ECA(nn.Module):
    def __init__(self, input_channel, layers, num_classes):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64
        super(MSResNet_ECA, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)       
        # Add ECA after initial convolution
        self.eca_init = ECA(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 3x3 branch layers
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.eca3x3_1 = ECA(64)  # ECA after first 3x3 block     
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.eca3x3_2 = ECA(128)  # ECA after second 3x3 block   
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.eca3x3_3 = ECA(256)  # ECA after third 3x3 block  
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        # 5x5 branch layers
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.eca5x5_1 = ECA(64)  # ECA after first 5x5 block      
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.eca5x5_2 = ECA(128)  # ECA after second 5x5 block
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.eca5x5_3 = ECA(256)  # ECA after third 5x5 block
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        # 7x7 branch layers
        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.eca7x7_1 = ECA(64)  # ECA after first 7x7 block 
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.eca7x7_2 = ECA(128)  # ECA after second 7x7 block  
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.eca7x7_3 = ECA(256)  # ECA after third 7x7 block 
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)
        
        ## Add a final ECA after feature fusion
        #self.final_eca = ECA(256*3)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1280*3, num_classes)  
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.eca_init(x0)  # Apply ECA after initial conv
        x0 = self.maxpool(x0)

        # 3x3 branch with ECA after each block
        x = self.layer3x3_1(x0)
        x = self.eca3x3_1(x)  # Apply ECA after first block  
        x = self.layer3x3_2(x)
        x = self.eca3x3_2(x)  # Apply ECA after second block   
        x = self.layer3x3_3(x)
        x = self.eca3x3_3(x)  # Apply ECA after third block   
        x = self.maxpool3(x)

        # 5x5 branch with ECA after each block
        y = self.layer5x5_1(x0)
        y = self.eca5x5_1(y)  # Apply ECA after first block 
        y = self.layer5x5_2(y)
        y = self.eca5x5_2(y)  # Apply ECA after second block  
        y = self.layer5x5_3(y)
        y = self.eca5x5_3(y)  # Apply ECA after third block  
        y = self.maxpool5(y)

        # 7x7 branch with ECA after each block
        z = self.layer7x7_1(x0)
        z = self.eca7x7_1(z)  # Apply ECA after first block  
        z = self.layer7x7_2(z)
        z = self.eca7x7_2(z)  # Apply ECA after second block   
        z = self.layer7x7_3(z)
        z = self.eca7x7_3(z)  # Apply ECA after third block   
        z = self.maxpool7(z)

        # Concatenate features from all branches
        out = torch.cat([x, y, z], dim=1)
        
        ## Apply final ECA to the fused features
        #out = self.final_eca(out)
        
        # Flatten and classify
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.drop(out)
        out1 = self.fc(out)
        out1 = self.sigmoid(out1)

        return out1

model = MSResNet_ECA(input_channel=num_channels, layers=[1,1,1,1], num_classes=num_classes).to(device)

# ## 3. Training and validation

import torch.nn.functional as F
import torch.nn.init as init
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# He Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize biases to zero
        
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
model.apply(initialize_weights)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.6)

num_epochs=30
train_loss, train_losses = [], []
train_acc, train_accuracies = [], []
val_loss, val_losses = [], [] 
val_acc, val_accuracies = [], []

#==============Training loop====================
for epoch in range(num_epochs):
    train_start_time = time.time()
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for X, y in train_loader:
        X, y = X.to(device).float(), y.to(device).float()
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward() # Backward pass 
        optimizer.step()  
        running_loss += loss.item() * X.size(0)
        predicted = (outputs > 0.5).float() 
        train_correct += (predicted == y).all(dim=1).sum().item()
        train_total += y.size(0)
    
    scheduler.step() #warm-up LR

    train_end_time = time.time() 
    train_time = train_end_time - train_start_time 

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_acc = train_correct / train_total
    train_accuracies.append(train_acc) 
    
#=============Validation=======================
    val_start_time = time.time()
    model.eval()
    running_loss = 0.0
    val_correct = 0
    val_total = 0 
    with torch.no_grad():
        for X, y in train_loader:
            X, y = X.to(device).float(), y.to(device).float()
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)            
            predicted = (outputs > 0.5).float() 
            val_correct += (predicted == y).all(dim=1).sum().item()
            val_total += y.size(0)
            
    val_end_time = time.time()  
    val_time = val_end_time - val_start_time 
                
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = val_correct / val_total
    val_accuracies.append(val_acc)          
        
        
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Time: {train_time:.2f}s | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {val_time:.2f}s')

# ## 4. Testing

from sklearn.metrics import classification_report, accuracy_score

test_sig_no = 200 
test_data_folder = os.path.join(os.getcwd(), "FIF_test_data")

def evaluate_on_snr_levels(model, criterion, batch_size=120, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate model on all SNR levels using the saved pickle files"""
    class_names = ['Flicker', 'Swell', 'Sag', 'Interruption', 'Harmonics', 'Oscillatory transient']
    results = {}
    
    for snr in snr_levels:
        pickle_path = os.path.join(test_data_folder, f"test_{snr}.pkl")
        
        # Check if pickle file exists
        if not os.path.exists(pickle_path):
            print(f"Error: Test data pickle for {snr} not found at {pickle_path}")
            print("Please run prepare_test_datasets() first.")
            continue        
        # Load test data from pickle
        with open(pickle_path, 'rb') as f:
            test_dataset = pickle.load(f)        
        X_test, y_test = test_dataset.X, test_dataset.y
        dataset = PQDDataset(X_test, y_test)
        dataloader = DataLoader(dataset, batch_size=120, shuffle=False)
        
        # Testing
        test_start_time = time.time()
        model.eval()      
        y_true_list = []
        y_pred_list = []
        test_loss, test_losses = [], []
        test_acc, test_accuracies = [], []
        running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Testing loop
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()              
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)               
                predicted = (outputs > 0.5).float()
                
                y_true_list.append(y_batch.cpu().numpy())
                y_pred_list.append(predicted.cpu().numpy())             
                test_correct += (predicted == y_batch).all(dim=1).sum().item()
                test_total += y_batch.size(0)
        
        # Calculate metrics
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        test_loss = running_loss / len(dataloader.dataset)
        test_acc = test_correct / test_total
        
        # Convert lists to numpy arrays
        y_true_np = np.vstack(y_true_list)
        y_pred_np = np.vstack(y_pred_list)
        
        ## Generate classification report
        report = classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0, output_dict=True)
        
        # Print results
        print(f'\nSNR: {snr}')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Time: {test_time:.2f}s')
        print("\nClassification Report:")
        print(classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0))
        
    return results

results = evaluate_on_snr_levels(model, criterion)
