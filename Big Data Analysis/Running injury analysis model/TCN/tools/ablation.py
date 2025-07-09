import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from models.TCN import TCN
from data_process.dataloader import InjuryDataset
from torch.utils.data import DataLoader
from data_process.ablation_feature_structure import create_samples
from data_process.load_data import Load_data
from data_process.data_cleaning import process_dumplicated_data, process_null_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from models.focal_loss import FocalLoss
from visualization.visualize import plot_training_metrics
from logs.log_param import count_parameters
from data_process.post_process import log_epoch_predictions, count_predictions, count_predictions_by_epoch
from data_process.data_balance_process import balance_sequence_data

# Load the data
day_data, week_data = Load_data()
# Process the data
day_data, week_data = process_dumplicated_data(day_data, week_data)
day_data, week_data = process_null_data(day_data, week_data)

# If you do not want to use a feature, set its corresponding position to True
X_seq, X_weekly, y = create_samples(
    day_data,
    mask_seq_feats=[False, False, False, False, True]
)

X_seq_train, X_seq_temp, X_weekly_train, X_weekly_temp, y_train, y_temp = train_test_split(
    X_seq, X_weekly, y, test_size=0.4, random_state=42
)
X_seq_train, X_weekly_train, y_train = balance_sequence_data(
    X_seq_train, X_weekly_train, y_train, method='smote_enn', random_state=42
)
X_seq_val, X_seq_test, X_weekly_val, X_weekly_test, y_val, y_test = train_test_split(
    X_seq_temp, X_weekly_temp, y_temp, test_size=0.5, random_state=42
)

X_seq_train = torch.tensor(X_seq_train).float()
X_weekly_train = torch.tensor(X_weekly_train).float()
y_train = torch.tensor(y_train).float()

X_seq_val = torch.tensor(X_seq_val).float()
X_weekly_val = torch.tensor(X_weekly_val).float()
y_val = torch.tensor(y_val).float()

X_seq_test = torch.tensor(X_seq_test).float()
X_weekly_test = torch.tensor(X_weekly_test).float()
y_test = torch.tensor(y_test).float()

num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
pos_weight_val = num_neg / max(num_pos, 1)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
print(f"Train Set: Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight_val}")

num_pos = (y_val == 1).sum()
num_neg = (y_val == 0).sum()
pos_weight_val = num_neg / max(num_pos, 1)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
print(f"Val Set: Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight_val}")

num_pos = (y_test == 1).sum()
num_neg = (y_test == 0).sum()
pos_weight_val = num_neg / max(num_pos, 1)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
print(f"Test Set: Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight_val}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pos_weight = pos_weight.to(device)

# Define the model
model = TCN(
    input_size=5,
    output_size=1,
    num_channels=[64, 64, 64],
    kernel_size=3,
    dropout=0.2
)

criterion = FocalLoss(alpha=0.25, gamma=2.0)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = InjuryDataset(X_seq_train, X_weekly_train, y_train)
val_dataset = InjuryDataset(X_seq_val, X_weekly_val, y_val)
test_dataset = InjuryDataset(X_seq_test, X_weekly_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

best_val_f1 = 0

metrics_history = {
    'train_loss': [], 'val_loss': [],
    'train_precision': [], 'val_precision': [],
    'train_recall': [], 'val_recall': [],
    'train_f1': [], 'val_f1': [],
    'train_auc': [], 'val_auc': []
}

for epoch in range(20):
    # ========== Training ==========
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []

    for x_seq, x_weekly, y in train_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        optimizer.zero_grad()
        x_seq = x_seq.permute(0, 2, 1)
        logits = model(x_seq)
        loss = criterion(logits, y.unsqueeze(1).float())
        loss.backward()
        total_loss += loss.item() * y.size(0)
        optimizer.step()

        # Calculate accuracy
        probs = torch.sigmoid(logits).detach().cpu()
        preds = (probs > 0.5).int()
        all_logits.append(probs)
        all_preds.append(preds)
        all_labels.append(y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    train_loss = total_loss / len(train_loader.dataset)
    train_precision = precision_score(all_labels, all_preds, zero_division=0)
    train_recall = recall_score(all_labels, all_preds, zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, zero_division=0)
    train_auc = roc_auc_score(all_labels, all_logits)
    log_epoch_predictions(all_preds, all_labels, epoch, save_path="logs/ablation_epoch_prediction_log.csv")


    # ========== Validation ==========
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for x_seq, x_weekly, y in val_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            x_seq = x_seq.permute(0, 2, 1)
            logits = model(x_seq)
            loss = criterion(logits, y.unsqueeze(1).float())
            val_loss += loss.item() * y.size(0)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).int()
            val_preds.append(preds)
            val_labels.append(y.cpu())

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    val_loss /= len(val_loader.dataset)
    val_precision = precision_score(val_labels, val_preds, zero_division=0)
    val_recall = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    with torch.no_grad():
        preds = np.concatenate([
            torch.sigmoid(model(x_seq.to(device).permute(0, 2, 1))).cpu().numpy()
            for x_seq, _, _ in val_loader
        ])
    val_auc = roc_auc_score(val_labels, preds)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
    print(f"            Val Loss={val_loss:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
    log_epoch_predictions(val_preds, val_labels, epoch, save_path="logs/ablation_val_epoch_prediction_log.csv")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "ablation_saved_model/best_model.pth")

    metrics_history['train_loss'].append(train_loss)
    metrics_history['train_precision'].append(train_precision)
    metrics_history['train_recall'].append(train_recall)
    metrics_history['train_f1'].append(train_f1)
    metrics_history['train_auc'].append(train_auc)

    metrics_history['val_loss'].append(val_loss)
    metrics_history['val_precision'].append(val_precision)
    metrics_history['val_recall'].append(val_recall)
    metrics_history['val_f1'].append(val_f1)
    metrics_history['val_auc'].append(val_auc)

# ========== Testing ==========
model.eval()
test_loss = 0
test_preds = []
test_labels = []
with torch.no_grad():
    for x_seq, x_weekly, y in test_loader:
        x_seq, y = x_seq.to(device), y.to(device)
        x_seq = x_seq.permute(0, 2, 1)
        logits = model(x_seq)
        loss = criterion(logits, y.unsqueeze(1).float())
        test_loss += loss.item() * y.size(0)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > 0.5).int()
        test_preds.append(preds)
        test_labels.append(y.cpu())

test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()
test_loss /= len(test_loader.dataset)
test_precision = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)
with torch.no_grad():
        preds = np.concatenate([
            torch.sigmoid(model(x_seq.to(device).permute(0, 2, 1))).cpu().numpy()
            for x_seq, _, _ in test_loader
        ])
test_auc = roc_auc_score(test_labels, preds)
print(f"Model Parameters: {count_parameters(model)}")
print(f"Test Loss={test_loss:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
log_epoch_predictions(test_preds, test_labels, 0, save_path="logs/ablation_test_prediction_log.csv")

log_path = "logs/ablation_epoch_prediction_log.csv"
#result = count_predictions(log_path)
#print("Training Predictions Count:")
#for key, value in result.items():
    #print(f"{key}: {value}")
#print(count_predictions_by_epoch(log_path))


log_path = "logs/ablation_val_epoch_prediction_log.csv"
#result = count_predictions(log_path)
#print("Validation Predictions Count:")
#for key, value in result.items():
    #print(f"{key}: {value}")
#print(count_predictions_by_epoch(log_path))

log_path = "logs/ablation_test_prediction_log.csv"
#result = count_predictions(log_path)
#print("Test Predictions Count:")
#for key, value in result.items():
    #print(f"{key}: {value}")
#print(count_predictions_by_epoch(log_path))

# Plot training metrics
plot_training_metrics(metrics_history)
