#!/usr/bin/env python3
"""
scripts/05_plot_figures.py
================================================================================
Reproduce all publication figures (Figures 1–3 and Supplementary Figure 1) for:
"Objective Identification of Basal, Primed, and Active Conformational States in 
Wild-Type Human MuSK Kinase via K-ESPRIT" (Computers in Biology and Medicine, 2026).

Usage:
    python scripts/05_plot_figures.py
    python scripts/05_plot_figures.py --data path/to/processed_metrics_with_plddt.csv --outdir outputs/figures
================================================================================
"""

import argparse
import os
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# ==============================================================================
# グローバル描画設定（Nature / Elsevier 論文基準のタイポグラフィ）
# ==============================================================================
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.color"] = "#222222"
plt.rcParams["ytick.color"] = "#222222"
plt.rcParams["figure.dpi"] = 300

# カラーパレットの統一定義
VAR_COLORS = {
    "WT": "#3B6790",    # クラシックブルー
    "S752D": "#4DAF4A", # グリーン
    "D754A": "#E66101"  # オレンジ
}

STATE_COLORS = {
    "Closed Basin": "#3B6790",        # スレートブルー
    "Intermediate Cluster": "#E0A93B",# ゴールド／マスタード
    "Open Basin": "#C0503E"           # テラコッタ／クリムゾン
}

VARIANTS_ORDER = ["WT", "S752D", "D754A"]
STATES_ORDER = ["Closed Basin", "Intermediate Cluster", "Open Basin"]


# ==============================================================================
# 1. Figure 1: 統計的収束性、A-loop柔軟性、およびラッチ結合の完全性
# ==============================================================================
def plot_figure_1(df, outdir):
    print("--> Plotting Figure 1 (Convergence, RMSD, and Latch integrity)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # --- 1A: モンテカルロ・ブートストラップ収束解析 (n=1,000) ---
    sample_sizes = [25, 50, 75, 100]
    markers = {"WT": "o", "S752D": "s", "D754A": "D"}
    np.random.seed(42)

    for v in VARIANTS_ORDER:
        sub = df[df["Variant"] == v]
        # 開放状態の定義: RMSD > 10.0 Å かつ RS2–RS3 <= 4.5 Å
        active_ind = ((sub["A_loop_RMSD"] > 10.0) & (sub["RS2_RS3_dist"] <= 4.5)).astype(int).values
        means, ci_lows, ci_highs = [], [], []
        for N in sample_sizes:
            boot_samples = np.random.choice(active_ind, size=(1000, N), replace=True).mean(axis=1) * 100
            means.append(np.mean(boot_samples))
            ci_lows.append(np.percentile(boot_samples, 2.5))
            ci_highs.append(np.percentile(boot_samples, 97.5))

        axes[0].plot(sample_sizes, means, color=VAR_COLORS[v], marker=markers[v], label=f"{v} Mean", lw=2, markersize=6)
        axes[0].fill_between(sample_sizes, ci_lows, ci_highs, color=VAR_COLORS[v], alpha=0.18, label=f"{v} 95% CI")

    axes[0].set_xlabel("Ensemble Sample Size (N)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Active State Population (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-5, 105)
    axes[0].set_xticks(sample_sizes)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="center left", fontsize=8.5, framealpha=0.9)
    axes[0].set_title("(A)", loc="left", fontweight="bold", fontsize=14)

    # --- 1B: A-loop Cα-RMSD 分布 ---
    sns.violinplot(ax=axes[1], data=df, x="Variant", y="A_loop_RMSD", order=VARIANTS_ORDER,
                   palette=VAR_COLORS, inner="quartile", cut=0, linewidth=1.2)
    sns.stripplot(ax=axes[1], data=df, x="Variant", y="A_loop_RMSD", order=VARIANTS_ORDER,
                  color="black", alpha=0.35, size=4, jitter=0.2)
    axes[1].axhline(10.0, color="darkred", linestyle="--", lw=1.5, label="Fully Open (>10.0 Å)")
    axes[1].set_xlabel("Variant", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("A-loop Cα-RMSD vs. WT #27 (Å)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(-0.5, 18.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[1].set_title("(B)", loc="left", fontweight="bold", fontsize=14)

    # --- 1C: D754–Q801 自己抑制ラッチ距離（cut=0 で外挿を防ぐ） ---
    sns.violinplot(ax=axes[2], data=df, x="Variant", y="Latch_dist", order=VARIANTS_ORDER,
                   palette=VAR_COLORS, inner="quartile", cut=0, linewidth=1.2)
    sns.stripplot(ax=axes[2], data=df, x="Variant", y="Latch_dist", order=VARIANTS_ORDER,
                  color="black", alpha=0.35, size=4, jitter=0.2)
    axes[2].axhline(3.5, color="darkred", linestyle="--", lw=1.5, label="H-bond Limit (3.5 Å)")
    axes[2].set_xlabel("Variant", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Min Heavy-Atom Distance (Å)", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, 31)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[2].set_title("(C)", loc="left", fontweight="bold", fontsize=14)

    plt.tight_layout()
    fig.savefig(outdir / "Figure1.png", dpi=300)
    fig.savefig(outdir / "Figure1.pdf")
    plt.close(fig)
    print("  ✓ Figure 1 saved (PNG & PDF).")


# ==============================================================================
# 2. Figure 2: 配座状態分割、二面角幾何、および2次元確率ランドスケープ
# ==============================================================================
def plot_figure_2(df, outdir):
    print("--> Plotting Figure 2 (State partitioning, Ramachandran, and 2D Landscape)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- 2A: 状態分割の積み上げ棒グラフ ---
    ct = pd.crosstab(df["Variant"], df["Conformational_State"], normalize="index") * 100
    ct = ct.reindex(index=VARIANTS_ORDER, columns=STATES_ORDER).fillna(0)

    bottom = np.zeros(len(VARIANTS_ORDER))
    for s in STATES_ORDER:
        values = ct[s].values
        label_text = s.replace(" Basin", "").replace(" Cluster", "")
        bars = axes[0, 0].bar(VARIANTS_ORDER, values, bottom=bottom, color=STATE_COLORS[s],
                              label=label_text, edgecolor="black", width=0.55, lw=1.0)
        for bar, val, b in zip(bars, values, bottom):
            if val >= 4.0:
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2, b + val / 2,
                                f"{int(round(val))}%", ha="center", va="center",
                                color="white", fontweight="bold", fontsize=10.5)
        bottom += values

    axes[0, 0].set_xlabel("Variant", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Ensemble Population (%)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    axes[0, 0].set_title("(A)", loc="left", fontweight="bold", fontsize=14)

    # --- 2B: DFG-Phe744 ラマチャンドランプロット (Φ vs. Ψ) ---
    rect_beta = patches.Rectangle((-180, 20), 130, 140, linewidth=0, facecolor="#FDF3D0",
                                  alpha=0.6, label="Favored β / Extended Region")
    rect_alpha = patches.Rectangle((-120, -100), 90, 90, linewidth=0, facecolor="#E5ECF2",
                                   alpha=0.6, label="Favored α-Helix Region")
    axes[0, 1].add_patch(rect_beta)
    axes[0, 1].add_patch(rect_alpha)

    for s in STATES_ORDER:
        sub = df[df["Conformational_State"] == s]
        axes[0, 1].scatter(sub["phi_F"], sub["psi_F"], color=STATE_COLORS[s], label=s,
                           alpha=0.75, edgecolors="black", linewidth=0.5, s=35)

    axes[0, 1].set_xlim(-180, 180)
    axes[0, 1].set_ylim(-180, 180)
    axes[0, 1].set_xlabel("DFG-Phe744 Φ [degrees]", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("DFG-Phe744 Ψ [degrees]", fontsize=11, fontweight="bold")
    axes[0, 1].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0, 1].axvline(0, color="gray", linestyle=":", alpha=0.5)
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)
    axes[0, 1].legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    axes[0, 1].set_title("(B)", loc="left", fontweight="bold", fontsize=14)

    # --- 2C: 2次元反応座標散布図 ---
    for v in VARIANTS_ORDER:
        sub = df[df["Variant"] == v]
        axes[1, 0].scatter(sub["A_loop_RMSD"], sub["RS2_RS3_dist"], color=VAR_COLORS[v],
                           label=v, alpha=0.6, edgecolors="none", s=32)

    axes[1, 0].axhline(4.5, color="darkred", linestyle="--", lw=1.5, label="R-spine Packing Limit (4.5 Å)")
    axes[1, 0].axvline(10.0, color="gray", linestyle=":", lw=1.0)
    # 実験結晶ベンチマーク 9SN0 (WT)
    axes[1, 0].plot(0.45, 6.52, marker="*", markersize=14, color="gold", markeredgecolor="black", label="9SN0 (WT Crystal)")
    # 実験結晶ベンチマーク 9SNU (S752D)
    axes[1, 0].axhline(3.15, color="teal", linestyle=":", lw=1.5, label="9SNU Core (3.15 Å)")

    # 代表メドイドのハイライト（実測値）
    medoids = {
        "WT #27 (Closed)": (0.00, 6.78),
        "WT #45 (Intermediate)": (2.99, 3.02),
        "WT #37 (Open)": (12.44, 3.08)
    }
    for name, (x_m, y_m) in medoids.items():
        axes[1, 0].plot(x_m, y_m, marker="o", markersize=11, markeredgecolor="red", markeredgewidth=2.2, fillstyle="none")
        axes[1, 0].text(x_m, y_m + 0.25, name, fontsize=8.5, fontweight="bold", color="darkred", ha="center")

    axes[1, 0].text(14.0, 6.0, "Sterically Forbidden Zone\n(0/300 models)", ha="center", va="center",
                    color="gray", fontsize=9.5, style="italic")
    axes[1, 0].set_xlabel("A-loop Cα-RMSD vs. WT #27 (Å)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("RS2–RS3 Distance (Å)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(-0.5, 18.5)
    axes[1, 0].set_ylim(2.0, 7.5)
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)
    axes[1, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    axes[1, 0].set_title("(C)", loc="left", fontweight="bold", fontsize=14)

    # --- 2D: ガウスカーネル密度推定（KDE）等高線 ---
    sns.kdeplot(ax=axes[1, 1], data=df[df["Variant"] == "WT"], x="A_loop_RMSD", y="RS2_RS3_dist",
                cmap="Blues", fill=True, alpha=0.45, levels=6)
    sns.kdeplot(ax=axes[1, 1], data=df[df["Variant"] == "D754A"], x="A_loop_RMSD", y="RS2_RS3_dist",
                cmap="Reds", fill=True, alpha=0.45, levels=6)
    axes[1, 1].axhline(4.5, color="darkred", linestyle="--", lw=1.5)

    # 2段階活性化アノテーション矢印
    axes[1, 1].annotate("", xy=(2.99, 3.15), xytext=(0.43, 6.5),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
    axes[1, 1].text(1.2, 5.0, "Step 1: Local R-spine\nPacking (RS2–RS3)",
                    color="crimson", fontweight="bold", fontsize=9.5)

    axes[1, 1].annotate("", xy=(12.5, 3.1), xytext=(3.5, 3.1),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
    axes[1, 1].text(7.5, 2.45, "Step 2: Full A-loop\nOutward Extension",
                    color="crimson", fontweight="bold", fontsize=9.5)

    axes[1, 1].set_xlabel("A-loop Cα-RMSD vs. WT #27 (Å)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("RS2–RS3 Distance (Å)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlim(-0.5, 18.5)
    axes[1, 1].set_ylim(2.0, 7.5)
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)
    axes[1, 1].set_title("(D)", loc="left", fontweight="bold", fontsize=14)

    plt.tight_layout()
    fig.savefig(outdir / "Figure2.png", dpi=300)
    fig.savefig(outdir / "Figure2.pdf")
    plt.close(fig)
    print("  ✓ Figure 2 saved (PNG & PDF).")


# ==============================================================================
# 3. Figure 3: 触媒塩橋の堅牢性と制御スパインの段階的疎水性集合
# ==============================================================================
def plot_figure_3(df, outdir):
    print("--> Plotting Figure 3 (Catalytic salt-bridge & R-spine assembly)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- 3A: 触媒塩橋 vs クレフト呼吸運動の線形回帰 ---
    slope, intercept, r_val, p_val, _ = stats.linregress(df["Cleft_Width"], df["Salt_Bridge_dist"])
    sns.scatterplot(ax=axes[0, 0], data=df, x="Cleft_Width", y="Salt_Bridge_dist",
                    hue="Variant", hue_order=VARIANTS_ORDER, palette=VAR_COLORS, alpha=0.75, s=42)
    x_vals = np.linspace(df["Cleft_Width"].min(), df["Cleft_Width"].max(), 100)
    axes[0, 0].plot(x_vals, intercept + slope * x_vals, "k:", lw=1.5,
                    label=f"Linear Fit: $R^2={r_val**2:.4f}$ ($p={p_val:.4f}$, n.s.)")

    axes[0, 0].set_xlabel("Inter-Lobe Cleft Distance (Å)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("K609–E626 Distance (Å)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim(2.4, 3.6)
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0, 0].set_title("(A)", loc="left", fontweight="bold", fontsize=14)

    # --- 3B: K609–E626 塩橋の変異体別バイオリン図 ---
    sns.violinplot(ax=axes[0, 1], data=df, x="Variant", y="Salt_Bridge_dist", order=VARIANTS_ORDER,
                   palette=VAR_COLORS, inner="quartile", cut=0, linewidth=1.2)
    sns.stripplot(ax=axes[0, 1], data=df, x="Variant", y="Salt_Bridge_dist", order=VARIANTS_ORDER,
                  color="black", alpha=0.35, size=4, jitter=0.2)
    axes[0, 1].axhline(3.5, color="darkred", linestyle="--", lw=1.5, label="Electrostatic Limit (3.5 Å)")
    axes[0, 1].set_xlabel("Variant", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("K609–E626 Distance (Å)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim(2.4, 3.6)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 1].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0, 1].set_title("(B)", loc="left", fontweight="bold", fontsize=14)

    # --- 3C: 状態別 R-spine 3ペア箱ひげ図（★査読指摘対応：Spine_Label削除 & 明確な凡例） ---
    spine_df = df.melt(
        id_vars=["Conformational_State"],
        value_vars=["RS3_RS4_dist", "RS2_RS3_dist", "RS1_RS2_dist"],
        var_name="Spine_Pair",
        value_name="Distance"
    )
    spine_map = {
        "RS3_RS4_dist": "RS3–RS4\n(M630–L641)",
        "RS2_RS3_dist": "RS2–RS3\n(F744–M630)",
        "RS1_RS2_dist": "RS1–RS2\n(H723–F744)"
    }
    spine_df["Spine_Pair"] = spine_df["Spine_Pair"].map(spine_map)

    sns.boxplot(
        ax=axes[1, 0],
        data=spine_df,
        x="Spine_Pair",
        y="Distance",
        hue="Conformational_State",
        hue_order=STATES_ORDER,
        palette=STATE_COLORS,
        width=0.6,
        fliersize=3.5,
        linewidth=1.1
    )
    axes[1, 0].axhline(4.5, color="darkred", linestyle="--", lw=1.5, label="Packing Limit (4.5 Å)")
    axes[1, 0].set_xlabel("")  # Spine_Label の完全消去
    axes[1, 0].set_ylabel("Min Heavy-Atom Distance (Å)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylim(2.0, 8.2)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    # パネル内に明示的な凡例ボックスを表示
    axes[1, 0].legend(title="Conformational State", loc="upper left", fontsize=9, title_fontsize=9.5, framealpha=0.95)
    axes[1, 0].set_title("(C)", loc="left", fontweight="bold", fontsize=14)

    # --- 3D: WT 代表メドイド3種の重原子間距離棒グラフ ---
    medoid_data = {
        "RS3–RS4\n(M630–L641)": [4.10, 3.99, 4.01],
        "RS2–RS3\n(F744–M630)": [6.78, 3.02, 3.08],
        "RS1–RS2\n(H723–F744)": [7.13, 3.85, 4.02]
    }
    x_pairs = list(medoid_data.keys())
    x = np.arange(len(x_pairs))
    width = 0.22

    rects1 = axes[1, 1].bar(x - width, [medoid_data[p][0] for p in x_pairs], width,
                            label="WT #27 (Closed)", color=STATE_COLORS["Closed Basin"], edgecolor="black", lw=1.0)
    rects2 = axes[1, 1].bar(x, [medoid_data[p][1] for p in x_pairs], width,
                            label="WT #45 (Intermediate)", color=STATE_COLORS["Intermediate Cluster"], edgecolor="black", lw=1.0)
    rects3 = axes[1, 1].bar(x + width, [medoid_data[p][2] for p in x_pairs], width,
                            label="WT #37 (Open)", color=STATE_COLORS["Open Basin"], edgecolor="black", lw=1.0)

    # 各バーの上に実測数値（Å）を表示
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            axes[1, 1].annotate(f"{height:.2f} Å",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    axes[1, 1].axhline(4.5, color="darkred", linestyle="--", lw=1.5, label="Packing Limit (4.5 Å)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(x_pairs, fontsize=9.5)
    axes[1, 1].set_ylabel("Min Heavy-Atom Distance (Å)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylim(0, 8.5)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 1].legend(loc="upper left", fontsize=9, framealpha=0.95)
    axes[1, 1].set_title("(D)", loc="left", fontweight="bold", fontsize=14)

    plt.tight_layout()
    fig.savefig(outdir / "Figure3.png", dpi=300)
    fig.savefig(outdir / "Figure3.pdf")
    plt.close(fig)
    print("  ✓ Figure 3 saved (PNG & PDF).")


# ==============================================================================
# 4. Supplement Figure 1: pLDDT 信頼度プロファイルと DFG モチーフの秩序化
# ==============================================================================
def plot_supp_figure_1(df, outdir):
    print("--> Plotting Supplement Figure 1 (pLDDT profiles)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # データ整形（Melt形式）
    plddt_region_df = df.melt(
        id_vars=["Variant", "Conformational_State", "A_loop_RMSD"],
        value_vars=["Core_pLDDT", "DFG_pLDDT", "A_loop_pLDDT"],
        var_name="Region",
        value_name="pLDDT"
    )
    region_map = {
        "Core_pLDDT": "C-lobe Core\n(αE / αF)",
        "DFG_pLDDT": "DFG Motif\n(743–745)",
        "A_loop_pLDDT": "Activation Loop\n(743–765)"
    }
    plddt_region_df["Region"] = plddt_region_df["Region"].map(region_map)

    # --- S1A: 変異体別 pLDDT 分布 ---
    sns.boxplot(
        ax=axes[0, 0],
        data=plddt_region_df,
        x="Region",
        y="pLDDT",
        hue="Variant",
        hue_order=VARIANTS_ORDER,
        palette=VAR_COLORS,
        width=0.6,
        fliersize=3.5,
        linewidth=1.1
    )
    axes[0, 0].axhline(70.0, color="green", linestyle="--", lw=1.2, label="High Confidence (70)")
    axes[0, 0].axhline(50.0, color="orange", linestyle=":", lw=1.2, label="Low Confidence (50)")
    axes[0, 0].set_xlabel("")  # Region_Label 削除
    axes[0, 0].set_ylabel("Mean Cα pLDDT Score", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim(35, 102)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    axes[0, 0].set_title("(A)", loc="left", fontweight="bold", fontsize=14)

    # --- S1B: 状態別 pLDDT 分布 ---
    sns.boxplot(
        ax=axes[0, 1],
        data=plddt_region_df,
        x="Region",
        y="pLDDT",
        hue="Conformational_State",
        hue_order=STATES_ORDER,
        palette=STATE_COLORS,
        width=0.6,
        fliersize=3.5,
        linewidth=1.1
    )
    axes[0, 1].axhline(70.0, color="green", linestyle="--", lw=1.2, label="High Confidence (70)")
    axes[0, 1].axhline(50.0, color="orange", linestyle=":", lw=1.2, label="Low Confidence (50)")
    axes[0, 1].set_xlabel("")  # Region_Label 削除
    axes[0, 1].set_ylabel("Mean Cα pLDDT Score", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim(35, 102)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 1].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    axes[0, 1].set_title("(B)", loc="left", fontweight="bold", fontsize=14)

    # --- S1C: A-loop pLDDT vs. RMSD 散布図 ---
    sns.scatterplot(
        ax=axes[1, 0],
        data=df,
        x="A_loop_RMSD",
        y="A_loop_pLDDT",
        hue="Variant",
        hue_order=VARIANTS_ORDER,
        palette=VAR_COLORS,
        alpha=0.75,
        s=38
    )
    axes[1, 0].axvline(10.0, color="darkred", linestyle="--", lw=1.2, label="Open Threshold (10.0 Å)")
    axes[1, 0].axhline(50.0, color="orange", linestyle=":", lw=1.2, label="Low Confidence Cutoff (50)")
    axes[1, 0].set_xlabel("A-loop Cα-RMSD vs. WT #27 (Å)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("A-loop Mean Cα pLDDT Score", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlim(-0.5, 18.5)
    axes[1, 0].set_ylim(35, 75)
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)
    axes[1, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    axes[1, 0].set_title("(C)", loc="left", fontweight="bold", fontsize=14)

    # --- S1D: DFGモチーフ pLDDT バイオリン図と Mann-Whitney U 検定 ---
    sns.violinplot(
        ax=axes[1, 1],
        data=df,
        x="Conformational_State",
        y="DFG_pLDDT",
        order=STATES_ORDER,
        palette=STATE_COLORS,
        inner="quartile",
        cut=0,
        linewidth=1.2
    )
    sns.stripplot(
        ax=axes[1, 1],
        data=df,
        x="Conformational_State",
        y="DFG_pLDDT",
        order=STATES_ORDER,
        color="black",
        alpha=0.35,
        size=4,
        jitter=0.2
    )
    axes[1, 1].axhline(70.0, color="green", linestyle="--", lw=1.2, label="High Confidence (70)")
    axes[1, 1].axhline(50.0, color="orange", linestyle=":", lw=1.2, label="Low Confidence (50)")

    # 統計検定アノテーションボックス
    axes[1, 1].text(
        0.5, 0.90,
        "Closed vs. Open DFG pLDDT:\nMann–Whitney U $p = 3.33 \\times 10^{-42}$",
        transform=axes[1, 1].transAxes,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.95)
    )
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("DFG Motif (743–745) pLDDT Score", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylim(45, 80)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    axes[1, 1].set_title("(D)", loc="left", fontweight="bold", fontsize=14)

    plt.tight_layout()
    fig.savefig(outdir / "Supplement_Figure1.png", dpi=300)
    fig.savefig(outdir / "Supplement_Figure1.pdf")
    plt.close(fig)
    print("  ✓ Supplement Figure 1 saved (PNG & PDF).")


# ==============================================================================
# メイン実行ルーチン
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Figures 1-3 & Supp Fig 1 for MuSK K-ESPRIT.")
    parser.add_argument("--data", type=str, default="", help="Path to processed_metrics_with_plddt.csv")
    parser.add_argument("--outdir", type=str, default="outputs/figures", help="Directory to save figures")
    args = parser.parse_args()

    # 入力CSVの探索パス
    if args.data:
        csv_path = Path(args.data)
    else:
        candidate_paths = [
            Path("data/processed_metrics_with_plddt.csv"),
            Path("../data/processed_metrics_with_plddt.csv"),
            Path("/content/MuSK-K-ESPRIT/data/processed_metrics_with_plddt.csv"),
            Path("/content/processed_metrics_with_plddt.csv")
        ]
        csv_path = None
        for p in candidate_paths:
            if p.exists():
                csv_path = p
                break

    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "processed_metrics_with_plddt.csv が見つかりません。--data でパスを指定してください。"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("K-ESPRIT Publication Figure Reproduction Pipeline")
    print(f"  Input Dataset : {csv_path.resolve()}")
    print(f"  Output Folder : {outdir.resolve()}")
    print("=================================================================")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} structural model records.")

    plot_figure_1(df, outdir)
    plot_figure_2(df, outdir)
    plot_figure_3(df, outdir)
    plot_supp_figure_1(df, outdir)

    print("\n=================================================================")
    print("All manuscript figures successfully reproduced and saved!")
    print(f"Output files located at: {outdir.resolve()}")
    print("=================================================================")


if __name__ == "__main__":
    main()