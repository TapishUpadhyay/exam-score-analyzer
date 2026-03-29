"""
analysis.py
============
Exam Score Analyzer & Feedback Generator
-----------------------------------------
Pipeline:
  1. Generate synthetic student data  →  data/student_scores.csv
  2. Exploratory Data Analysis        →  saves plots to data/
  3. K-Means Clustering               →  model/kmeans_model.pkl
  4. Feedback Generation              →  data/feedback_report.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECTS       = ["Mathematics", "Science", "English", "History", "Computer_Science"]
DATA_PATH      = "data/student_scores.csv"
FEEDBACK_PATH  = "data/feedback_report.csv"
MODEL_PATH     = "model/kmeans_model.pkl"
SCALER_PATH    = "model/scaler.pkl"
WEAK_THRESHOLD = 50
N_STUDENTS     = 200
SEED           = 42

os.makedirs("data",  exist_ok=True)
os.makedirs("model", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate_data():
    np.random.seed(SEED)
    high   = int(N_STUDENTS * 0.30)
    medium = int(N_STUDENTS * 0.45)
    low    = N_STUDENTS - high - medium

    def scores(mean, std, n):
        return np.clip(np.random.normal(mean, std, n), 0, 100).astype(int)

    data = {s: np.concatenate([scores(82, 8, high), scores(60, 10, medium), scores(38, 10, low)])
            for s in SUBJECTS}

    data["Student_ID"]   = [f"STU{str(i).zfill(4)}" for i in range(1, N_STUDENTS + 1)]
    data["Attendance_%"] = np.concatenate([
        np.clip(np.random.normal(90, 5,  high),   60, 100),
        np.clip(np.random.normal(75, 8,  medium),  50, 100),
        np.clip(np.random.normal(55, 10, low),     30, 100),
    ]).astype(int)

    df = pd.DataFrame(data)
    df["Total"]   = df[SUBJECTS].sum(axis=1)
    df["Average"] = (df["Total"] / len(SUBJECTS)).round(2)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[1/4] Data generated → {DATA_PATH}  ({len(df)} students)")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def run_eda(df):
    # Score distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, subj in enumerate(SUBJECTS):
        axes[i].hist(df[subj], bins=20, color="steelblue", edgecolor="white")
        axes[i].set_title(f"{subj.replace('_',' ')} Distribution", fontsize=11)
        axes[i].set_xlabel("Score"); axes[i].set_ylabel("Students")
    axes[-1].hist(df["Average"], bins=20, color="coral", edgecolor="white")
    axes[-1].set_title("Overall Average", fontsize=11)
    axes[-1].set_xlabel("Score"); axes[-1].set_ylabel("Students")
    plt.suptitle("Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/score_distributions.png", dpi=150); plt.close()

    # Subject averages
    avg    = df[SUBJECTS].mean().sort_values()
    colors = ["#e74c3c" if v < 60 else "#f39c12" if v < 75 else "#2ecc71" for v in avg]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(avg.index, avg.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, avg.values):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=10)
    plt.xlabel("Average Score"); plt.xlim(0, 110)
    plt.title("Subject-wise Class Average", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/subject_averages.png", dpi=150); plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[SUBJECTS + ["Attendance_%"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/correlation_heatmap.png", dpi=150); plt.close()

    # Attendance vs Average
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Attendance_%"], df["Average"], alpha=0.5,
                color="darkorchid", edgecolors="white")
    plt.xlabel("Attendance (%)"); plt.ylabel("Average Score")
    plt.title("Attendance vs Average Score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/attendance_vs_average.png", dpi=150); plt.close()

    print("[2/4] EDA complete → 4 plots saved in data/")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════

def cluster_students(df):
    features = df[SUBJECTS + ["Attendance_%"]]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Label by average score
    cluster_means = df.groupby("Cluster")["Average"].mean().sort_values(ascending=False)
    label_map = {cluster_means.index[0]: "High Performer",
                 cluster_means.index[1]: "Average Performer",
                 cluster_means.index[2]: "At-Risk"}
    df["Performance_Group"] = df["Cluster"].map(label_map)

    # Save model + scaler
    with open(MODEL_PATH,  "wb") as f: pickle.dump(kmeans, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    # Cluster distribution plot
    counts = df["Performance_Group"].value_counts()
    colors = {"High Performer": "#2ecc71", "Average Performer": "#f39c12", "At-Risk": "#e74c3c"}
    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values,
                   color=[colors[g] for g in counts.index], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", fontsize=11)
    plt.title("Performance Group Distribution", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig("data/cluster_distribution.png", dpi=150); plt.close()

    # Radar chart
    group_means = df.groupby("Performance_Group")[SUBJECTS].mean()
    N      = len(SUBJECTS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for group, row in group_means.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, linewidth=2, label=group, color=colors[group])
        ax.fill(angles, vals, alpha=0.1, color=colors[group])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.replace("_", " ") for s in SUBJECTS], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Subject Radar by Group", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("data/cluster_radar.png", dpi=150); plt.close()

    print(f"[3/4] Clustering done → model saved in model/")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEEDBACK GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate_feedback(row):
    weak   = [s for s in SUBJECTS if row[s] < WEAK_THRESHOLD]
    strong = [s for s in SUBJECTS if row[s] >= 75]
    group  = row["Performance_Group"]

    lines = [
        f"Student ID    : {row['Student_ID']}",
        f"Performance   : {group}",
        f"Average Score : {row['Average']:.1f} / 100",
        f"Attendance    : {row['Attendance_%']}%",
        "",
    ]
    if   group == "High Performer":     lines.append("Excellent work! You are among the top performers.")
    elif group == "Average Performer":  lines.append("Steady progress. Focused effort can push you higher.")
    else:                               lines.append("You are currently at risk. Immediate attention needed.")

    if strong: lines.append(f"Strong subjects : {', '.join(s.replace('_',' ') for s in strong)}")
    if weak:
        lines.append(f"Weak subjects   : {', '.join(s.replace('_',' ') for s in weak)}")
        lines.append("  -> Allocate extra time and seek instructor help.")
    if   row["Attendance_%"] < 60: lines.append("Attendance is critically low — attend classes regularly.")
    elif row["Attendance_%"] < 75: lines.append("Try to improve attendance for better results.")

    return "\n".join(lines)


def generate_all_feedback(df):
    df["Feedback"] = df.apply(generate_feedback, axis=1)
    df[["Student_ID", "Performance_Group", "Average", "Attendance_%"] + SUBJECTS + ["Feedback"]]\
        .to_csv(FEEDBACK_PATH, index=False)
    print(f"[4/4] Feedback report saved → {FEEDBACK_PATH}")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Exam Score Analyzer & Feedback Generator")
    print("=" * 55)
    df = generate_data()
    df = run_eda(df)
    df = cluster_students(df)
    df = generate_all_feedback(df)
    print("=" * 55)
    print("  Pipeline complete! Check data/ and model/")
    print("=" * 55)
