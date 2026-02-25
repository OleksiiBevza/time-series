######################################################################################
########################## DATA PREPROCESSING ########################################
######################################################################################

# packages for dealing with data
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# packages for modelling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, balanced_accuracy_score

# packages for modelling online
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)



# ------------------------------------------------------
# 1. EXTRACT RELEVANT VARIABLES
# ------------------------------------------------------
def extract_variables(x_sar_csv: str | Path) -> pd.DataFrame:
    """
    Function selects variables that are stated in a paper as significant 
    for modelling purposes

    Summing disk and network columns comes from the fact that we want to 
    see the total work the server is doing across all parts
    
    Multiplying by 512 we make bytes so that it is clear how much data i
    moving

    In `interface_util_pct` take the max cause single disk hitting 100% can 
    stop an app, while disks doing nothing
    """
    df = pd.read_csv(x_sar_csv)

    # disk data in seconds
    bytes_read_per_s = df.filter(regex=r"^dev\d+-\d+_rd_sec/s$").sum(axis=1) * 512
    bytes_written_per_s = df.filter(regex=r"^dev\d+-\d+_wr_sec/s$").sum(axis=1) * 512

    # network traffic in seconds
    rx_packets_per_s = df.filter(regex=r"^eth\d+_rxpck/s$").sum(axis=1)
    tx_packets_per_s = df.filter(regex=r"^eth\d+_txpck/s$").sum(axis=1)
    rx_kb_per_s = df.filter(regex=r"^eth\d+_rxkB/s$").sum(axis=1)
    tx_kb_per_s = df.filter(regex=r"^eth\d+_txkB/s$").sum(axis=1)

    interface_util_pct = df.filter(regex=r"^dev\d+-\d+_%util$").max(axis=1)

    return pd.DataFrame(
        {
            "TimeStamp": df["TimeStamp"],

            # CPU metrics: idle, work, waiting
            "cpu_idle_pct": df["all_%%idle"],
            "cpu_user_pct": df["all_%%usr"],
            "cpu_system_pct": df["all_%%sys"],
            "cpu_iowait_pct": df["all_%%iowait"],

            # emory and swap in kb
            "mem_used_kb": df["kbmemused"],
            "mem_committed_kb": df["kbcommit"],
            "swap_used_kb": df["kbswpused"],
            "swap_cached_kb": df["kbswpcad"],

            # transaction and block rates
            "read_trans_per_s": df["rtps"],
            "write_trans_per_s": df["wtps"],
            "bytes_read_per_s": bytes_read_per_s,
            "bytes_written_per_s": bytes_written_per_s,
            "block_reads_per_s": df["bread/s"],
            "block_writes_per_s": df["bwrtn/s"],

            # operating system performance metrics
            "new_processes_per_s": df["proc/s"],
            "context_switches_per_s": df["cswch/s"],

            # aggregated network stats
            "rx_packets_per_s": rx_packets_per_s,
            "tx_packets_per_s": tx_packets_per_s,
            "rx_kb_per_s": rx_kb_per_s,
            "tx_kb_per_s": tx_kb_per_s,
            "interface_util_pct": interface_util_pct,
        }
    )




# ------------------------------------------------------
# 2. DEFINE LABELS
# ------------------------------------------------------
def make_labels_from_Y(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Function labels data wen FPS dropped below 20.
    If yes = 1, if no = 0
    """
    # make copy of original data
    y = Y.copy()
    # check every second to see if the FPS dropped below 20
    # and label it 
    y["fps_violation"] = (y["DispFrames"] < 20).astype(int)
    return y[["TimeStamp", "fps_violation"]]







# ------------------------------------------------------
# 3. DEFINE LABELS
# ------------------------------------------------------
def make_task_dataset(root: str | Path, scenario: str, W: int, H: int):
    """
    Function creates dataset by giving every metric a label like in 
    'make_labels_from_Y'
    """
    root = Path(root)

    # put together hardware metrics (X) and quality labels (Y)
    # extract only variables for modelling
    X = extract_variables(root / "X_SAR" / scenario / "X.csv")
    Y = pd.read_csv(root / "X_SAR" / scenario / "Y.csv")[["TimeStamp", "DispFrames", "noAudioPlayed"]]
    labels = make_labels_from_Y(Y)

    # put together that every hardware metric has label
    df = X.merge(labels, on="TimeStamp", how="inner")
    # make it numpy
    X_feat = df.drop(columns=["TimeStamp", "fps_violation"]).to_numpy()
    inc = df["fps_violation"].to_numpy()

    # W = look back history
    # H = look ahead prediction
    X_windows, y_targets, t_idx = [], [], []
    n = len(df)
    for t in range(W - 1, n - H):
        # if system failing now = skip
        # we wanna predict comming crash from good performance
        if inc[t] == 1:
            continue
        # take last W seconds as input 
        X_windows.append(X_feat[t - W + 1 : t + 1])
        # look ahead H seconds. If any failure give score 1
        # cause failure will occur
        y_targets.append(int(inc[t + 1 : t + H + 1].max()))
        t_idx.append(t)

    return np.asarray(X_windows), np.asarray(y_targets), np.asarray(t_idx)




# ------------------------------------------------------
# 4. TRAIN - VALIDATE - TEST SETS
# ------------------------------------------------------
def time_split_indices(n: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Function defines train, validate, test sets
    """
    # how many samples go into each bucket
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # past data to train model
    train_idx = np.arange(0, n_train)
    # validation set 
    val_idx = np.arange(n_train, n_train + n_val)
    # test set
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx




# ------------------------------------------------------
# 5. TRAIN - VALIDATE - TEST SETS
# ------------------------------------------------------
def flatten_windows(X_windows: np.ndarray) -> np.ndarray:
    """
    Takes (Number of samples, Time window, Features) and 
    reshapes them 
    """
    # X_windows: (N, W, F) -> (N, W*F)
    return X_windows.reshape(X_windows.shape[0], -1)




# ------------------------------------------------------
# 6. RESHAPE FEATURES
# ------------------------------------------------------
def summary_features(X_windows: np.ndarray) -> np.ndarray:
    """"
    Function takes (Number of samples, Time window, Features)
    and reshapes to (Number of samples, 5 * Features)
    """
    # X_windows: (N, W, F) -> (N, 5*F) using mean,std,min,max,last over W
    # basic stats across the time dimension (axis=1)
    mean = X_windows.mean(axis=1)
    std  = X_windows.std(axis=1)
    mn   = X_windows.min(axis=1)
    mx   = X_windows.max(axis=1)
    # take recent value in the window to capture current state
    last = X_windows[:, -1, :]
    # shape them as: (N, 5 * Features)
    return np.concatenate([mean, std, mn, mx, last], axis=1)




# ------------------------------------------------------
# 7. DEFINE THRESHOLD
# ------------------------------------------------------
def choose_threshold_fpr_cap(p_val, y_val, fpr_cap=0.01):
    """"
    Function calculates confusion matrix and defines the 
    best possible threshold.
    """
    # make inputs numpy
    p_val = np.asarray(p_val)
    y_val = np.asarray(y_val).astype(int)

    best_thr = float(p_val.max())
    best_recall = -1.0

    for thr in np.unique(p_val):
        y_hat = (p_val >= thr).astype(int)
        # calculat confusion matrix
        TP = ((y_hat == 1) & (y_val == 1)).sum()    # true positives
        FP = ((y_hat == 1) & (y_val == 0)).sum()    # false positives
        FN = ((y_hat == 0) & (y_val == 1)).sum()    # false negatives
        TN = ((y_hat == 0) & (y_val == 0)).sum()    # true negatives

        # if false positive rate (FP / total negatives) > 1%
        #ignore threshold 
        if FP / (FP + TN) > fpr_cap:
            continue

        # if threshold is good:  false positive rate < 1%
        # try to catch more positives (Recall) than  previous
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        if recall > best_recall:
            best_recall = recall
            best_thr = float(thr)

    return best_thr




# ------------------------------------------------------
# 8. CALCULATE FALSE POSITIVE RATE
# ------------------------------------------------------
def fpr_at_threshold(p, y, thr):
    """
    Function calculates false positive rate
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    # binary predictions (1 if prob > 1% threshold, else 0)
    y_hat = (p >= thr).astype(int)
    FP = ((y_hat == 1) & (y == 0)).sum()    # false positives
    TN = ((y_hat == 0) & (y == 0)).sum()    # true negatives
    # calculate false positive rate
    return FP / (FP + TN) if (FP + TN) else 0.0




# ------------------------------------------------------
# 9. EVALUATION
# ------------------------------------------------------
def evaluate_simple(p_test, y_test, thr):
    """
    Function evaluates results of the models
    """
    pr_auc = average_precision_score(y_test, p_test)
    y_hat = (p_test >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)
    ba = balanced_accuracy_score(y_test, y_hat)
    return pr_auc, precision, recall, f1, ba







# ------------------------------------------------------
# 10. ONLINE TREE CLASS
# ------------------------------------------------------
# ============================================================
# ADAPTIVE THRESHOLD USING ACTUAL ONLINE DECISIONS
# ============================================================

class AdaptiveThreshold:
    """
    Adaptive threshold that tries to keep FPR near a target, using
    the *actual* binary decisions made online (not re-thresholded
    probabilities).
    """
    def __init__(
        self,
        initial_threshold: float = 0.5,
        fpr_target: float = 0.01,
        window_size: int = 200,
        adjustment_factor: float = 1.05,
        min_threshold: float = 0.2,
        max_threshold: float = 0.9,
        fpr_tolerance: float = 0.002,  # don't overreact to tiny deviations
    ):
        self.threshold = initial_threshold
        self.fpr_target = fpr_target
        self.window_size = window_size
        self.adjustment_factor = adjustment_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.fpr_tolerance = fpr_tolerance

        # Store actual online decisions and labels
        self.recent_y_pred = deque(maxlen=window_size)
        self.recent_y_true = deque(maxlen=window_size)

    def get_threshold(self) -> float:
        return self.threshold

    def update(self, y_pred: int, y_true: int) -> float:
        """
        Update threshold based on *actual* decision y_pred and true label y_true.
        """
        self.recent_y_pred.append(int(y_pred))
        self.recent_y_true.append(int(y_true))

        # Need some data before adapting
        if len(self.recent_y_true) < self.window_size // 2:
            return self.threshold

        y_pred_arr = np.array(self.recent_y_pred)
        y_true_arr = np.array(self.recent_y_true)

        fp = ((y_pred_arr == 1) & (y_true_arr == 0)).sum()
        tn = ((y_pred_arr == 0) & (y_true_arr == 0)).sum()

        if fp + tn > 0:
            current_fpr = fp / (fp + tn)
        else:
            current_fpr = 0.0

        fpr_error = current_fpr - self.fpr_target

        # Only adjust if error is non-trivial
        if abs(fpr_error) > self.fpr_tolerance:
            if fpr_error > 0:  # FPR too high → be stricter
                self.threshold = min(
                    self.threshold * self.adjustment_factor, self.max_threshold
                )
            else:              # FPR too low → be more permissive
                self.threshold = max(
                    self.threshold / self.adjustment_factor, self.min_threshold
                )

        return self.threshold


# ============================================================
# OAUE-LIKE ENSEMBLE WITH WEIGHTED VOTING + WINDOW ACCURACY
# ============================================================

class OAUEEnsemble:
    """
    OAUE-style ensemble with:
      - Window-based per-tree accuracy.
      - Weighted voting based on recent accuracy.
      - Replacement of only the worst tree when accuracy < acc_min.
      - Replacement trees are trained on the current sample.
    """
    def __init__(
        self,
        n_estimators: int = 20,
        grace_period: int = 100,
        delta: float = 1e-2,
        tau: float = 0.05,
        acc_init: float = 0.75,
        acc_min: float = 0.55,
        acc_window_size: int = 200,
        vote_temperature: float = 5.0,  # controls how sharp weights depend on accuracy
    ):
        self.n_estimators = n_estimators
        self.acc_init = float(acc_init)
        self.acc_min = float(acc_min)
        self.acc_window_size = int(acc_window_size)
        self.vote_temperature = float(vote_temperature)

        self._tree_kwargs = dict(
            grace_period=grace_period,
            delta=delta,
            tau=tau,
        )

        from river import tree
        self.models = [
            tree.HoeffdingTreeClassifier(**self._tree_kwargs)
            for _ in range(n_estimators)
        ]

        # Per-tree sliding windows of correctness (1 = correct, 0 = incorrect)
        self.correct_windows = [
            deque(maxlen=self.acc_window_size)
            for _ in range(n_estimators)
        ]

    @staticmethod
    def _row_to_dict(x_row):
        """Convert numpy array / dict into river's {feature: value} dict."""
        if isinstance(x_row, dict):
            return {str(k): float(v) for k, v in x_row.items()}

        if hasattr(x_row, "shape"):
            x_row = np.ravel(x_row)
            return {str(i): float(v) for i, v in enumerate(x_row)}

        return {str(i): float(v) for i, v in enumerate(np.atleast_1d(x_row))}

    def _get_accuracy(self, idx: int) -> float:
        """Window-based accuracy for model idx."""
        w = self.correct_windows[idx]
        if not w:
            return self.acc_init
        return float(sum(w) / len(w))

    def _get_all_accuracies(self):
        return [self._get_accuracy(i) for i in range(self.n_estimators)]

    def _get_weights(self):
        """
        Turn accuracies into voting weights via a softmax-like transform.
        """
        accuracies = np.array(self._get_all_accuracies(), dtype=float)
        # Stabilize:
        a_centered = accuracies - np.mean(accuracies)
        logits = self.vote_temperature * a_centered
        weights = np.exp(logits)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return np.ones_like(weights) / len(weights)
        return weights / weights_sum

    def predict_proba_one(self, x_row) -> float:
        """
        Return probability of positive class as weighted fraction of trees
        voting 1. Final decision in your pipeline is done by comparing this
        probability to the adaptive threshold.
        """
        x = self._row_to_dict(x_row)

        votes = []
        for model in self.models:
            proba = model.predict_proba_one(x)
            p1 = proba.get(1, 0.5)
            y_pred = 1 if p1 >= 0.5 else 0
            votes.append(y_pred)

        votes = np.array(votes, dtype=float)
        weights = self._get_weights()

        # Weighted fraction of positive votes
        p_hat = float(np.clip(np.sum(weights * votes), 0.01, 0.99))
        return p_hat

    def learn_one(self, x_row, y_true: int):
        """
        Update ensemble with one labeled example.

        Steps:
          1. Each tree predicts, updates its correctness window, then learns.
          2. Compute window accuracies.
          3. Replace worst tree if its accuracy < acc_min.
             The new tree is immediately trained on the current sample.
        """
        x = self._row_to_dict(x_row)
        y_true = int(y_true)

        from river import tree

        # 1) Update all trees, record correctness, learn from sample
        for i, model in enumerate(self.models):
            proba = model.predict_proba_one(x)
            p1 = proba.get(1, 0.5)
            y_pred = 1 if p1 >= 0.5 else 0

            correct = int(y_pred == y_true)
            self.correct_windows[i].append(correct)

            model.learn_one(x, y_true)

        # 2) Find worst-performing tree
        accuracies = np.array(self._get_all_accuracies(), dtype=float)
        worst_idx = int(np.argmin(accuracies))

        # 3) Replace worst tree if its accuracy is too low
        if accuracies[worst_idx] < self.acc_min:
            # New tree
            new_tree = tree.HoeffdingTreeClassifier(**self._tree_kwargs)
            # Train replacement tree on current sample (as requested)
            new_tree.learn_one(x, y_true)

            self.models[worst_idx] = new_tree
            self.correct_windows[worst_idx].clear()  # start its accuracy window fresh


# ============================================================
# PURE PREQUENTIAL EVALUATION
# ============================================================

def evaluate_online_prequential(
    X_train,
    y_train,
    X_eval,
    y_eval,
    timestamps,
    model_class,
    model_params,
    fpr_target: float = 0.01,
    window_size: int = 200,
    initial_threshold: float = 0.5,
    adjustment_factor: float = 1.05,
):
    """
    Pure prequential evaluation:

      1. Train model on (X_train, y_train).
      2. For each sample in eval stream:
         - predict probability
         - make binary decision with current threshold
         - update model with true label
         - update threshold using the *actual* decision and label
    """
    # ----- model init -----
    if model_class == SGDClassifier:
        classes = np.array([0, 1])
        cw = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        cw_dict = {c: w for c, w in zip(classes, cw)}

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            learning_rate="optimal",
            class_weight=cw_dict,
            random_state=42,
        )
    else:
        model = model_class(**model_params)

    # ----- threshold init -----
    threshold_adapter = AdaptiveThreshold(
        initial_threshold=initial_threshold,
        fpr_target=fpr_target,
        window_size=window_size,
        adjustment_factor=adjustment_factor,
        min_threshold=0.2,
        max_threshold=0.9,
    )

    # ----- offline training on train set -----
    if model_class == SGDClassifier:
        batch_size = 32
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            model.partial_fit(batch_X, batch_y, classes=classes)
    else:
        for xi, yi in zip(X_train, y_train):
            model.learn_one(xi, int(yi))

    # ----- prequential pass on eval stream -----
    probabilities = []
    binary_predictions = []
    thresholds = []

    for xi, yi_true, ts in zip(X_eval, y_eval, timestamps):
        # Predict probability
        if model_class == SGDClassifier:
            prob = float(model.predict_proba(xi.reshape(1, -1))[0, 1])
        else:
            prob = float(model.predict_proba_one(xi))

        thr = threshold_adapter.get_threshold()
        y_pred = 1 if prob >= thr else 0

        probabilities.append(prob)
        binary_predictions.append(y_pred)
        thresholds.append(thr)

        # Update model with label
        if model_class == SGDClassifier:
            model.partial_fit(xi.reshape(1, -1), [yi_true])
        else:
            model.learn_one(xi, int(yi_true))

        # IMPORTANT: update threshold using *actual* decision and label
        threshold_adapter.update(y_pred, int(yi_true))

    return {
        "probabilities": np.array(probabilities),
        "binary_predictions": np.array(binary_predictions),
        "thresholds": np.array(thresholds),
    }


# ------------------------------------------------------
# 11. VISUALIZATION 
# ------------------------------------------------------


def rolling_far_ba(y, yhat, x, window=500):
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    x = np.asarray(x)

    n = len(y)
    w = int(window)
    if w < 1:
        raise ValueError("window must be >= 1")

    tp = ((yhat == 1) & (y == 1)).astype(int)
    fp = ((yhat == 1) & (y == 0)).astype(int)
    tn = ((yhat == 0) & (y == 0)).astype(int)
    fn = ((yhat == 0) & (y == 1)).astype(int)

    def csum(a):
        return np.concatenate([[0], np.cumsum(a)])

    tp_c, fp_c, tn_c, fn_c = csum(tp), csum(fp), csum(tn), csum(fn)

    start = np.clip(np.arange(n) - w + 1, 0, n)
    end = np.arange(n) + 1

    TP = tp_c[end] - tp_c[start]
    FP = fp_c[end] - fp_c[start]
    TN = tn_c[end] - tn_c[start]
    FN = fn_c[end] - fn_c[start]

    FAR = FP / np.maximum(FP + TN, 1)
    TPR = TP / np.maximum(TP + FN, 1)
    TNR = TN / np.maximum(TN + FP, 1)
    BA  = 0.5 * (TPR + TNR)

    return x, FAR, BA


def plot_far_ba(x, far, ba, title="", fpr_cap=0.01):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # rolling FAR
    axes[0].plot(x, far)
    axes[0].axhline(fpr_cap, linestyle="--")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("False Positive Rate")
    axes[0].set_title(f"{title} rolling False positive rate")

    # rolling BA
    axes[1].plot(x, ba)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title(f"{title} rolling Balanced accuracy")

    plt.tight_layout()
    plt.show()



