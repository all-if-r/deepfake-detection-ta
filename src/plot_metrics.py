"""
Modul Visualisasi dan Analisis Metrik (Tugas 6 & 7)
Membaca file log dari `src/logs/` dan menghasilkan grafik publikasi:
1. Kurva Accuracy Training vs Validation
2. Kurva Loss Training vs Validation
3. Visualisasi Statistik GFS (Tahap 1, Tahap 2, Frekuensi Fallback)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_training_curves(history_csv: str = "src/logs/training_history.csv", output_dir: str = "src/logs"):
    """
    Tugas 7: Menghasilkan kurva Loss dan Accuracy Training vs Validation.
    """
    if not os.path.exists(history_csv):
        print(f"[Plot] File history tidak ditemukan: {history_csv}")
        return

    df = pd.read_csv(history_csv)
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Plot Loss
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2.2)
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.2, linestyle="--")
    plt.title("Kurva Loss Training vs Validation (Konfigurasi K5)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    loss_fig_path = os.path.join(output_dir, "loss_curve_k5.png")
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"[Plot] Kurva Loss disimpan ke: {loss_fig_path}")

    # 2. Plot Accuracy
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(df["epoch"], df["train_acc"] * 100, label="Training Accuracy", color="#2ca02c", linewidth=2.2)
    plt.plot(df["epoch"], df["val_acc"] * 100, label="Validation Accuracy", color="#d62728", linewidth=2.2, linestyle="--")
    plt.title("Kurva Akurasi Training vs Validation (Konfigurasi K5)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Akurasi (%)", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    acc_fig_path = os.path.join(output_dir, "accuracy_curve_k5.png")
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()
    print(f"[Plot] Kurva Akurasi disimpan ke: {acc_fig_path}")


def analyze_and_plot_gfs_statistics(gfs_stats_csv: str = "src/logs/gfs_statistics.csv", output_dir: str = "src/logs"):
    """
    Tugas 6: Meringkas statistik GFS dan membuat grafik distribusi.
    """
    if not os.path.exists(gfs_stats_csv):
        print(f"[Plot] File statistik GFS tidak ditemukan: {gfs_stats_csv}")
        return

    df = pd.read_csv(gfs_stats_csv)
    os.makedirs(output_dir, exist_ok=True)

    summary_stats = {
        "Total Video Dievaluasi": len(df),
        "Total Frame Dianalisis": df["total_frames"].sum(),
        "Rata-rata Frame/Video": df["total_frames"].mean(),
        "Total Frame Lolos Tahap 1 (ΔF)": df["passed_t1"].sum(),
        "Total Frame Gagal Tahap 1 (ΔF)": df["failed_t1"].sum(),
        "Rasio Lolos Tahap 1 (%)": (df["passed_t1"].sum() / max(1, df["total_frames"].sum())) * 100,
        "Total Frame Lolos Tahap 2 (Sharpness+YOLO)": df["passed_t2"].sum(),
        "Total Frame Gagal Tahap 2": df["failed_t2"].sum(),
        "Rasio Lolos Tahap 2 dari Tahap 1 (%)": (df["passed_t2"].sum() / max(1, df["passed_t1"].sum())) * 100,
        "Jumlah Video Terpicu Fallback": df["fallback_triggered"].sum(),
        "Persentase Video Terpicu Fallback (%)": (df["fallback_triggered"].sum() / max(1, len(df))) * 100
    }

    summary_df = pd.DataFrame(list(summary_stats.items()), columns=["Metrik", "Nilai"])
    summary_path = os.path.join(output_dir, "gfs_summary_table.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Plot] Tabel ringkasan GFS disimpan ke: {summary_path}")

    # Plot visualisasi perbandingan filtering GFS
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(7, 5), dpi=300)
    
    stages = ["Total Raw Frames", "Lolos Tahap 1 (ΔF)", "Lolos Tahap 2 (Sharpness+YOLO)"]
    counts = [df["total_frames"].sum(), df["passed_t1"].sum(), df["passed_t2"].sum()]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    bars = plt.bar(stages, counts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.01 * max(counts)), f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")

    plt.title("Statistik Penyaringan Golden Frame Selection (Tahap 1 & 2)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Jumlah Total Frame", fontsize=11)
    plt.tight_layout()

    gfs_chart_path = os.path.join(output_dir, "gfs_filtering_chart.png")
    plt.savefig(gfs_chart_path, dpi=300)
    plt.close()
    print(f"[Plot] Grafik statistik GFS disimpan ke: {gfs_chart_path}")

    return summary_df


if __name__ == "__main__":
    plot_training_curves()
    analyze_and_plot_gfs_statistics()
