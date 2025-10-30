import numpy as np
import torch

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt


def threshold_scan(video_scores,  gt_frames,save_path=None):
    thresholds = np.linspace(0, 1, 101)
    metrics = {"metric": [], "f1": [], "acc": [], "pre": [], "recall": []}

    best_metric, best_t = -1, 0.5

    frame_scores = np.repeat(video_scores, 16)[:len(gt_frames)]


    for t in thresholds:
        preds = (frame_scores > t).astype(int)

        TP = np.sum((preds == 1) & (gt_frames == 1))
        TN = np.sum((preds == 0) & (gt_frames == 0))
        FP = np.sum((preds == 1) & (gt_frames == 0))
        FN = np.sum((preds == 0) & (gt_frames == 1))

        if (TP+FN) == 0 or (TN+FP) == 0:
            metrics["metric"].append(np.nan)
            metrics["f1"].append(np.nan)
            metrics["acc"].append(np.nan)
            metrics["pre"].append(np.nan)
            metrics["recall"].append(np.nan)
            continue

        M = TP/(TP+FN) + TN/(TN+FP)
        metrics["metric"].append(M)


        metrics["f1"].append(f1_score(gt_frames, preds))
        metrics["acc"].append(accuracy_score(gt_frames, preds))
        metrics["pre"].append(precision_score(gt_frames, preds, zero_division=0))
        metrics["recall"].append(recall_score(gt_frames, preds))

        if M > best_metric:
            best_metric, best_t = M, t

    print(f"Best threshold = {best_t:.3f}, best_metric = {best_metric:.4f}")

    plt.figure(figsize=(8,6))
    for k,v in metrics.items():
        plt.plot(thresholds, v, label=k)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.axvline(x=best_t, color='red', linestyle='--', label=f"Best t={best_t:.2f}")
    plt.title("")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        print(f" {save_path}.png/.pdf/.svg")

    return best_t, best_metric, metrics



def find_best_threshold(video_scores, video_labels, save_path=None):

    best_t, best_metric = 0.5, -1
    metrics_curve = []
    for t in np.linspace(0, 1, 101):
        preds = (video_scores > t).astype(int)

        TP = np.sum((preds == 1) & (video_labels == 1))
        TN = np.sum((preds == 0) & (video_labels == 0))
        FP = np.sum((preds == 1) & (video_labels == 0))
        FN = np.sum((preds == 0) & (video_labels == 1))

        if (TP+FN) == 0 or (TN+FP) == 0:
            continue

        metric = TP/(TP+FN) + TN/(TN+FP)
        metrics_curve.append((t, metric))
        if metric > best_metric:
            best_metric = metric
            best_t = t

    if save_path is not None:
        np.save(save_path, np.array(metrics_curve))

    return best_t, best_metric,metrics_curve

def find_best_threshold_F1(video_scores, video_labels, save_path=None):

    best_t, best_metric = 0.5, -1
    metrics_curve = []
    for t in np.linspace(0, 1, 101):
        preds = (video_scores > t).astype(int)

        TP = np.sum((preds == 1) & (video_labels == 1))
        TN = np.sum((preds == 0) & (video_labels == 0))
        FP = np.sum((preds == 1) & (video_labels == 0))
        FN = np.sum((preds == 0) & (video_labels == 1))

        if (TP + FP) == 0 or (TP + FN) == 0:
            continue

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        metric = 2 * precision * recall / (precision + recall + 1e-8)

        metrics_curve.append((t, metric))
        if metric > best_metric:
            best_metric = metric
            best_t = t

    if save_path is not None:
        np.save(save_path, np.array(metrics_curve))

    return best_t, best_metric, metrics_curve



def load_threshold(path="/home/stu2023/jj/project/VAD_new_vision/best_threshold.pt"):
    ckpt = torch.load(path, map_location="cpu",weights_only=False)
    return ckpt['threshold']

def evaluate_frame_level(segment_scores, gt_frames, threshold, seg_len=16,save_path=None):
    metrics = { "f1": 0, "acc": 0, "pre": 0, "recall": 0 }


    frame_scores = np.repeat(segment_scores, seg_len)[:len(gt_frames)]


    preds = (frame_scores > threshold).astype(int)

    TP = np.sum((preds == 1) & (gt_frames == 1))
    TN = np.sum((preds == 0) & (gt_frames == 0))
    FP = np.sum((preds == 1) & (gt_frames == 0))
    FN = np.sum((preds == 0) & (gt_frames == 1))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    metrics["f1"] = float(f1)
    metrics["acc"] = float(acc)
    metrics["pre"] = float(precision)
    metrics["recall"] = float(recall)

    plot_metrics_bar(metrics, save_path=save_path)


    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc
    }

def plot_metrics_bar(metrics, save_path=None):
    keys = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6, 5))
    plt.bar(keys, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Metrics at Threshold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}.svg", bbox_inches="tight")
        print(f"图像已保存到 {save_path}.png/.pdf/.svg")

    # plt.show()