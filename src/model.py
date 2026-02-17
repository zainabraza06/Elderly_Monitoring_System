# src/model.py
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf
