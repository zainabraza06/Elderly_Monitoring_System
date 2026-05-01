import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, task_type="binary"):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if task_type != "binary" else torch.FloatTensor(labels)
        self.task_type = task_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLPModel(nn.Module):
    def __init__(self, input_size=384, hidden_sizes=None, num_classes=1, task="binary"):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)])
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.network(x)
        if self.task == "binary":
            return torch.sigmoid(out).squeeze()
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, num_classes=1, task="binary"):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.task = task

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        if self.task == "binary":
            return torch.sigmoid(out).squeeze()
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, num_classes=1, task="binary"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.task = task

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        if self.task == "binary":
            return torch.sigmoid(out).squeeze()
        return out


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, num_classes=1, task="binary"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.task = task

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.fc(context)
        if self.task == "binary":
            return torch.sigmoid(out).squeeze()
        return out


class CNNModel(nn.Module):
    def __init__(self, input_dim=3, num_classes=1, task="binary"):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(256 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.task = task

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        if self.task == "binary":
            return torch.sigmoid(out).squeeze()
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def apply_smote(X_train, y_train, task_type):
    X_flat = X_train.reshape(X_train.shape[0], -1)
    smote = SMOTE(random_state=42, sampling_strategy=0.5 if task_type == "binary" else "auto")
    X_bal, y_bal = smote.fit_resample(X_flat, y_train)
    X_bal = X_bal.reshape(-1, X_train.shape[1], X_train.shape[2])
    return X_bal, y_bal


def train_dl_model_full(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="binary",
    num_classes=1,
    epochs=20,
    model_name="Model",
    device=None,
):
    print(f"\n   Training {model_name} on {len(X_train):,} samples...")

    print("      Applying SMOTE to balance classes...")
    try:
        X_bal, y_bal = apply_smote(X_train, y_train, task_type)
        print(f"      After SMOTE: {len(X_bal):,} samples")
    except Exception as exc:
        print(f"      SMOTE failed: {exc}, using original data")
        X_bal, y_bal = X_train, y_train

    train_dataset = TimeSeriesDataset(X_bal, y_bal, task_type)
    test_dataset = TimeSeriesDataset(X_test, y_test, task_type)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    model = model.to(device)
    criterion = nn.BCELoss() if task_type == "binary" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024 * 1024)
    print(f"      Parameters: {num_params:,}, Size: {model_size_mb:.2f}MB")

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    start_time = time.time()
    best_f1 = 0
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            if task_type == "binary":
                labels = labels.float()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device)
                    outputs = model(data)
                    if task_type == "binary":
                        preds.extend((outputs.cpu().numpy() > 0.5).astype(int))
                    else:
                        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    trues.extend(labels.numpy())

            accuracy = accuracy_score(trues, preds)
            f1 = f1_score(trues, preds, average="weighted" if task_type != "binary" else "binary")

            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy

            print(
                "      Epoch {}/{}: Loss={:.4f}, Acc={:.4f}, F1={:.4f}".format(
                    epoch + 1,
                    epochs,
                    total_loss / len(train_loader),
                    accuracy,
                    f1,
                )
            )

            scheduler.step(loss)

    train_time = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_usage_mb = mem_after - mem_before

    model.eval()
    preds, trues = [], []
    inference_times = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            start_inf = time.perf_counter()
            outputs = model(data)
            end_inf = time.perf_counter()
            inference_times.append((end_inf - start_inf) / len(data) * 1000)

            if task_type == "binary":
                preds.extend((outputs.cpu().numpy() > 0.5).astype(int))
            else:
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.numpy())

    final_accuracy = accuracy_score(trues, preds)
    final_f1 = f1_score(trues, preds, average="weighted" if task_type != "binary" else "binary")
    inference_time_ms = np.mean(inference_times)

    seq_len, hidden_dim = 128, 128
    time_complexity = {
        "MLP": "O(n_features * n_neurons) ~= {:,} ops".format(384 * 256),
        "LSTM": "O(seq_len * hidden_dim^2) ~= {:,} ops".format(seq_len * hidden_dim**2),
        "BiLSTM": "O(seq_len * hidden_dim^2 * 2) ~= {:,} ops".format(
            seq_len * hidden_dim**2 * 2
        ),
        "BiLSTM+Att": "O(seq_len * hidden_dim^2 * 2 + seq_len) ~= {:,} ops".format(
            seq_len * hidden_dim**2 * 2 + seq_len
        ),
        "CNN": "O(seq_len * kernel_size * channels) ~= {:,} ops".format(seq_len * 3 * 256),
    }

    space_complexity = "O({:,} parameters) ~= {:.2f} MB".format(num_params, model_size_mb)

    print(f"      Final Results: Acc={final_accuracy:.4f}, F1={final_f1:.4f}")
    print(f"      Memory Usage: {memory_usage_mb:.1f}MB, Inference: {inference_time_ms:.3f}ms")

    return {
        "Model": model_name,
        "Accuracy": final_accuracy,
        "F1": final_f1,
        "Train_Time_s": train_time,
        "Inference_Time_ms": inference_time_ms,
        "Memory_Usage_MB": memory_usage_mb,
        "Model_Size_MB": model_size_mb,
        "Num_Params": num_params,
        "y_pred": preds,
        "y_true": trues,
        "Time_Complexity": time_complexity.get(model_name, "O(n^2)"),
        "Space_Complexity": space_complexity,
    }


def build_dl_models_fall():
    return [
        ("MLP", lambda: MLPModel(input_size=384, num_classes=1, task="binary")),
        ("LSTM", lambda: LSTMModel(num_classes=1, task="binary")),
        ("BiLSTM", lambda: BiLSTMModel(num_classes=1, task="binary")),
        ("BiLSTM+Att", lambda: BiLSTMAttentionModel(num_classes=1, task="binary")),
        ("CNN", lambda: CNNModel(num_classes=1, task="binary")),
    ]


def build_dl_models_adl(num_classes):
    return [
        ("MLP", lambda: MLPModel(input_size=384, num_classes=num_classes, task="multiclass")),
        ("LSTM", lambda: LSTMModel(num_classes=num_classes, task="multiclass")),
        ("BiLSTM", lambda: BiLSTMModel(num_classes=num_classes, task="multiclass")),
        (
            "BiLSTM+Att",
            lambda: BiLSTMAttentionModel(num_classes=num_classes, task="multiclass"),
        ),
        ("CNN", lambda: CNNModel(num_classes=num_classes, task="multiclass")),
    ]


def run_dl_experiments(
    X_train_raw,
    y_fall_train,
    X_test_raw,
    y_fall_test,
    X_train_adl,
    y_train_adl,
    X_test_adl,
    y_test_adl,
    epochs_fall=20,
    epochs_adl=25,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nTraining DL models with complete datasets:")
    print(f"   Fall Detection Training: {len(X_train_raw):,} samples")
    print(f"   Fall Detection Testing: {len(X_test_raw):,} samples")
    print(f"   ADL Classification Training: {len(X_train_adl):,} samples")
    print(f"   ADL Classification Testing: {len(X_test_adl):,} samples")

    print("\nTask 1: Fall Detection (DL with all data)")
    dl_fall_results = []
    for name, builder in build_dl_models_fall():
        result = train_dl_model_full(
            builder(),
            X_train_raw,
            y_fall_train,
            X_test_raw,
            y_fall_test,
            "binary",
            1,
            epochs_fall,
            name,
            device,
        )
        dl_fall_results.append(result)
        print(f"\n   {name} Summary:")
        print(f"      Accuracy: {result['Accuracy']:.4f}, F1: {result['F1']:.4f}")
        print(
            "      Train Time: {:.1f}s, Inference: {:.3f}ms".format(
                result["Train_Time_s"], result["Inference_Time_ms"]
            )
        )
        print(
            "      Model Size: {:.2f}MB, Memory: {:.1f}MB".format(
                result["Model_Size_MB"], result["Memory_Usage_MB"]
            )
        )
        print(f"      Parameters: {result['Num_Params']:,}")
        print(f"      Time Complexity: {result['Time_Complexity']}")
        print(f"      Space Complexity: {result['Space_Complexity']}")

    print("\nTask 2: ADL Classification (DL with all data)")
    dl_adl_results = []
    adl_classes = len(np.unique(y_train_adl))
    for name, builder in build_dl_models_adl(adl_classes):
        result = train_dl_model_full(
            builder(),
            X_train_adl,
            y_train_adl,
            X_test_adl,
            y_test_adl,
            "multiclass",
            adl_classes,
            epochs_adl,
            name,
            device,
        )
        dl_adl_results.append(result)
        print(f"\n   {name} Summary:")
        print(f"      Accuracy: {result['Accuracy']:.4f}, F1: {result['F1']:.4f}")
        print(
            "      Train Time: {:.1f}s, Inference: {:.3f}ms".format(
                result["Train_Time_s"], result["Inference_Time_ms"]
            )
        )
        print(
            "      Model Size: {:.2f}MB, Memory: {:.1f}MB".format(
                result["Model_Size_MB"], result["Memory_Usage_MB"]
            )
        )
        print(f"      Parameters: {result['Num_Params']:,}")

    return dl_fall_results, dl_adl_results
