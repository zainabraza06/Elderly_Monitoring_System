# src/eda.py
import numpy as np

def basic_eda(X_trials, y_trials):
    print("Total trials:", len(X_trials))
    print("Fall trials:", sum(y_trials))
    print("ADL trials:", len(y_trials) - sum(y_trials))

    lengths = [x.shape[0] for x in X_trials]
    print("Min length:", np.min(lengths))
    print("Max length:", np.max(lengths))
    print("Average length:", np.mean(lengths))

    print("Number of channels:", X_trials[0].shape[1])
