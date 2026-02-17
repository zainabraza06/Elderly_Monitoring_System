# main.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.loader import load_sisfall
from src.eda import basic_eda
from src.dataset_builder import build_dataset
from src.model import train_model
from src.evaluation import evaluate
DATA_PATH = r"C:\Users\User\Documents\4rth semester\AI\SisFall_dataset\data\SisFall_dataset"




# 1. Load
X_trials, y_trials, subjects = load_sisfall(DATA_PATH)

# 2. EDA
basic_eda(X_trials, y_trials)

# 3. Build window-level dataset
X, y = build_dataset(X_trials, y_trials)

# 4. Split (NO leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scale AFTER split
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train
model = train_model(X_train, y_train)

# 7. Evaluate
evaluate(model, X_test, y_test)
