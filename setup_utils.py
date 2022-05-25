import pandas as pd
import numpy as np
import torch
import os
import random
from sklearn.model_selection import KFold

SEED = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def label_fix(strng):
    strng = strng[1:-1]
    strip_lst = [" ", "."]
    outlst = [int(char) for char in strng if char not in strip_lst]
    assert len(outlst) == 7 or len(outlst) == 11
    return outlst


def doKfold(df_single, df_multi):
    seed_everything(seed=SEED)
    kf = KFold(
        n_splits=5, random_state=SEED, shuffle=True
    )  # Define the split - into 5 folds
    X = df_single.text.to_list()
    y = df_single.labels.to_list()
    multi_y = df_multi.labels.to_list()
    kf.get_n_splits(X)
    train_dfs, test_dfs = [], []

    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        y_train = [y[i] for i in train_index]

        X_test = [X[i] for i in test_index]
        y_test = [y[i] for i in test_index]

        multi_y_train = [multi_y[i] for i in train_index]
        multi_y_test = [multi_y[i] for i in test_index]

        train_df = pd.DataFrame(
            {"text": X_train, "sing_labels": y_train, "multi_labels": multi_y_train}
        )
        test_df = pd.DataFrame(
            {"text": X_test, "sing_labels": y_test, "multi_labels": multi_y_test}
        )
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    return train_dfs, test_dfs
