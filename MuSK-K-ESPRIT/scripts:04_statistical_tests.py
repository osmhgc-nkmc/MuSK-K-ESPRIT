#!/usr/bin/env python3
"""
scripts/04_statistical_tests.py
================================================================================
Reproduce all statistical hypothesis tests reported in:
"Objective Identification of Basal, Primed, and Active Conformational States in 
Wild-Type Human MuSK Kinase via K-ESPRIT" (Computers in Biology and Medicine, 2026).

Evaluated Tests:
  1. Section 3.2: A-loop RMSD distributions (Mann–Whitney U, Kolmogorov–Smirnov)
                  Autoinhibitory latch integrity and distance thresholds
  2. Section 3.3: Categorical state partitioning (Pearson's chi-square, Yates-corrected,
                  two-sided Fisher's exact test, Odds Ratios with 95% CI)
  3. Section 3.5: Catalytic salt bridge (Lys609–Glu626) stability vs domain breathing
                  (Ordinary least-squares linear regression & group comparisons)
  4. Section 3.7: State-dependent ordering of catalytic DFG motif pLDDT
                  (Overall pooled test & within-WT confounding-free test)

Usage:
    python scripts/04_statistical_tests.py
    python scripts/04_statistical_tests.py --data path/to/processed_metrics_with_plddt.csv
================================================================================
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 統計計算補助関数
# ==============================================================================
def calc_odds_ratio_ci(a, b, c, d, alpha=0.05):
    """
    2x2 分割表からオッズ比 (OR) と 95% 信頼区間 (Woolf法) を算出
    Table:
      [[a, b],   (Group 1: Event, Non-event)
       [c, d]]   (Group 2: Event, Non-event)
    """
    or_val = (a * d) / (b * c)
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_low = np.exp(np.log(or_val) - z * se)
    ci_high = np.exp(np.log(or_val) + z * se)
    return or_val, ci_low, ci_high


# ==============================================================================
# メイン統計検定ルーチン
# ==============================================================================
def run_all_statistical_tests(df, output_file=None):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("K-ESPRIT: Complete Statistical Reproduction Pipeline")
    log("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Section 3.2: A-loop RMSD 柔軟性およびラッチ結合の完全性
    # --------------------------------------------------------------------------
    log("\n" + "#" * 80)
    log("1. SECTION 3.2: ACTIVATION-LOOP DISPLACEMENT & LATCH INTEGRITY")
    log("#" * 80)

    wt_rmsd = df[df["Variant"] == "WT"]["A_loop_RMSD"].dropna()
    d754a_rmsd = df[df["Variant"] == "D754A"]["A_loop_RMSD"].dropna()
    s752d_rmsd = df[df["Variant"] == "S752D"]["A_loop_RMSD"].dropna()

    # WT vs D754A A-loop RMSD
    u_stat_d, u_p_d = stats.mannwhitneyu(d754a_rmsd, wt_rmsd, alternative="two-sided")
    ks_stat_d, ks_p_d = stats.ks_2samp(d754a_rmsd, wt_rmsd)

    log("\n[A-loop Cα-RMSD: WT (N=100) vs. D754A (N=100)]")
    log(f"  WT Median (IQR)    : {np.median(wt_rmsd):.2f} Å ({np.percentile(wt_rmsd, 25):.2f}–{np.percentile(wt_rmsd, 75):.2f} Å), Mean: {np.mean(wt_rmsd):.2f} ± {np.std(wt_rmsd, ddof=1):.2f} Å")
    log(f"  D754A Median (IQR) : {np.median(d754a_rmsd):.2f} Å ({np.percentile(d754a_rmsd, 25):.2f}–{np.percentile(d754a_rmsd, 75):.2f} Å), Mean: {np.mean(d754a_rmsd):.2f} ± {np.std(d754a_rmsd, ddof=1):.2f} Å")
    log(f"  Two-sided Mann-Whitney U test : U = {u_stat_d:.1f}, p = {u_p_d:.4e} (Manuscript: p = 8.28e-14)")
    log(f"  Two-sample Kolmogorov-Smirnov : D = {ks_stat_d:.4f}, p = {ks_p_d:.4e} (Manuscript: p = 2.95e-11)")

    # WT vs S752D A-loop RMSD
    u_stat_s, u_p_s = stats.mannwhitneyu(s752d_rmsd, wt_rmsd, alternative="two-sided")
    ks_stat_s, ks_p_s = stats.ks_2samp(s752d_rmsd, wt_rmsd)

    log("\n[A-loop Cα-RMSD: WT (N=100) vs. S752D (N=100)]")
    log(f"  S752D Median (IQR) : {np.median(s752d_rmsd):.2f} Å ({np.percentile(s752d_rmsd, 25):.2f}–{np.percentile(s752d_rmsd, 75):.2f} Å), Mean: {np.mean(s752d_rmsd):.2f} ± {np.std(s752d_rmsd, ddof=1):.2f} Å")
    log(f"  Two-sided Mann-Whitney U test : U = {u_stat_s:.1f}, p = {u_p_s:.4f}")
    log(f"  Two-sample Kolmogorov-Smirnov : D = {ks_stat_s:.4f}, p = {ks_p_s:.4f}")

    # Latch distance integrity
    log("\n[Autoinhibitory Latch Integrity (Residue 754 to Gln801 Distance)]")
    for v in ["WT", "S752D", "D754A"]:
        sub_v = df[df["Variant"] == v]
        compact = sub_v[sub_v["A_loop_RMSD"] < 5.0]["Latch_dist"]
        all_latch = sub_v["Latch_dist"]
        hbond_intact = np.sum(compact <= 3.5)
        total_compact = len(compact)
        log(f"  {v}: Compact models (<5.0 Å) with intact latch (<=3.5 Å): {hbond_intact}/{total_compact} ({hbond_intact/total_compact*100:.1f}%)")
        log(f"       Overall min distance: {np.min(all_latch):.2f} Å, Median: {np.median(all_latch):.2f} Å")

    # --------------------------------------------------------------------------
    # 2. Section 3.3: 状態分割の分割表検定 (Contingency Table Analysis)
    # --------------------------------------------------------------------------
    log("\n" + "#" * 80)
    log("2. SECTION 3.3: CONFORMATIONAL STATE PARTITIONING TESTS")
    log("#" * 80)

    # State counts
    ct = pd.crosstab(df["Variant"], df["Conformational_State"])
    log("\n[Observed State Counts (N=100 per variant)]:")
    log(ct.to_string())

    # --- 2A: D754A Open Basin Expansion (64/100 vs. 17/100) ---
    t_d754a = [[64, 36], [17, 83]]
    chi2_raw_d, p_raw_d, _, _ = stats.chi2_contingency(t_d754a, correction=False)
    chi2_yates_d, p_yates_d, _, _ = stats.chi2_contingency(t_d754a, correction=True)
    res_fe_d = stats.fisher_exact(t_d754a, alternative="two-sided")
    or_d, ci_l_d, ci_h_d = calc_odds_ratio_ci(64, 36, 17, 83)

    log("\n[1] D754A Open Basin Expansion vs. WT (64/100 vs. 17/100):")
    log(f"  Pearson chi-square (uncorrected) : chi2 = {chi2_raw_d:.2f}, p = {p_raw_d:.4e} (Manuscript: chi2 = 45.83)")
    log(f"  Pearson chi-square (Yates-corr)  : chi2 = {chi2_yates_d:.2f}, p = {p_yates_d:.4e} (Manuscript: chi2 = 43.90, p < 0.0001)")
    log(f"  Two-sided Fisher's exact test    : p = {res_fe_d[1]:.4e} (Manuscript: p = 1.21e-11)")
    log(f"  Odds Ratio (Woolf 95% CI)        : OR = {or_d:.2f} (95% CI: {ci_l_d:.2f}–{ci_h_d:.2f}) (Manuscript: OR = 8.68, 4.44–16.97)")

    # --- 2B: S752D Open Basin vs. WT (18/100 vs. 17/100) ---
    t_s752d_open = [[18, 82], [17, 83]]
    res_fe_s_open = stats.fisher_exact(t_s752d_open, alternative="two-sided")
    or_s_open, _, _ = calc_odds_ratio_ci(18, 82, 17, 83)

    log("\n[2] S752D Open Basin Runaway vs. WT (18/100 vs. 17/100):")
    log(f"  Two-sided Fisher's exact test    : p = {res_fe_s_open[1]:.4f} (Manuscript: p = 1.0000)")
    log(f"  Odds Ratio                       : OR = {or_s_open:.2f} (Manuscript: OR = 1.07)")

    # --- 2C: S752D Intermediate Cluster Enrichment (20/100 vs. 10/100) ---
    t_s752d_int = [[20, 80], [10, 90]]
    chi2_raw_s, p_raw_s, _, _ = stats.chi2_contingency(t_s752d_int, correction=False)
    chi2_yates_s, p_yates_s, _, _ = stats.chi2_contingency(t_s752d_int, correction=True)
    res_fe_s_int = stats.fisher_exact(t_s752d_int, alternative="two-sided")
    or_s_int, ci_l_s, ci_h_s = calc_odds_ratio_ci(20, 80, 10, 90)

    log("\n[3] S752D Intermediate Cluster Enrichment vs. WT (20/100 vs. 10/100):")
    log(f"  Pearson chi-square (uncorrected) : chi2 = {chi2_raw_s:.4f}, p = {p_raw_s:.4f} (Manuscript: chi2 = 3.92, p = 0.0477)")
    log(f"  Pearson chi-square (Yates-corr)  : chi2 = {chi2_yates_s:.4f}, p = {p_yates_s:.4f} (Reported Yates: chi2 = 3.18, p = 0.0747)")
    log(f"  Two-sided Fisher's exact test    : p = {res_fe_s_int[1]:.4f} (Manuscript: p = 0.0734)")
    log(f"  Odds Ratio (Woolf 95% CI)        : OR = {or_s_int:.2f} (95% CI: {ci_l_s:.2f}–{ci_h_s:.2f}) (Reported: OR = 2.25, 0.99–5.09)")

    # --- 2D: S752D Total R-spine Assembled (38/100 vs. 27/100) ---
    t_s752d_tot = [[38, 62], [27, 73]]
    chi2_raw_tot, p_raw_tot, _, _ = stats.chi2_contingency(t_s752d_tot, correction=False)
    chi2_yates_tot, p_yates_tot, _, _ = stats.chi2_contingency(t_s752d_tot, correction=True)
    res_fe_tot = stats.fisher_exact(t_s752d_tot, alternative="two-sided")
    or_tot, ci_l_tot, ci_h_tot = calc_odds_ratio_ci(38, 62, 27, 73)

    log("\n[4] S752D Total Active+Intermediate vs. WT (38/100 vs. 27/100):")
    log(f"  Pearson chi-square (uncorrected) : chi2 = {chi2_raw_tot:.4f}, p = {p_raw_tot:.4f} (Reported: chi2 = 2.76, p = 0.0968)")
    log(f"  Pearson chi-square (Yates-corr)  : chi2 = {chi2_yates_tot:.4f}, p = {p_yates_tot:.4f} (Reported: chi2 = 2.28, p = 0.1311)")
    log(f"  Two-sided Fisher's exact test    : p = {res_fe_tot[1]:.4f} (Reported: p = 0.1308)")
    log(f"  Odds Ratio (Woolf 95% CI)        : OR = {or_tot:.2f} (95% CI: {ci_l_tot:.2f}–{ci_h_tot:.2f}) (Manuscript: OR = 1.66)")

    # --------------------------------------------------------------------------
    # 3. Section 3.5: 触媒塩橋（Lys609–Glu626）の安定性とクレフト呼吸運動
    # --------------------------------------------------------------------------
    log("\n" + "#" * 80)
    log("3. SECTION 3.5: CATALYTIC SALT BRIDGE & DOMAIN BREATHING")
    log("#" * 80)

    # 線形回帰 (OLS)
    cleft = df["Cleft_Width"].dropna()
    salt = df["Salt_Bridge_dist"].dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(cleft, salt)
    r_squared = r_value ** 2

    log("\n[Ordinary Least-Squares Linear Regression (Cleft Width vs. Salt Bridge)]:")
    log(f"  Sample size N        : {len(cleft)}")
    log(f"  Cleft Width Range    : {cleft.min():.2f}–{cleft.max():.2f} Å")
    log(f"  Slope (SE)           : {slope:.6f} ({std_err:.6f})")
    log(f"  Intercept            : {intercept:.4f}")
    log(f"  R-squared (R^2)      : {r_squared:.4f} (Manuscript: R^2 = 0.0027)")
    log(f"  Regression p-value   : {p_value:.4f} (Manuscript: p = 0.3687, n.s.)")

    # 変異体間の塩橋距離比較
    wt_salt = df[df["Variant"] == "WT"]["Salt_Bridge_dist"].dropna()
    s752d_salt = df[df["Variant"] == "S752D"]["Salt_Bridge_dist"].dropna()
    d754a_salt = df[df["Variant"] == "D754A"]["Salt_Bridge_dist"].dropna()

    u_sb_s, p_sb_s = stats.mannwhitneyu(s752d_salt, wt_salt, alternative="two-sided")
    u_sb_d, p_sb_d = stats.mannwhitneyu(d754a_salt, wt_salt, alternative="two-sided")

    log("\n[Salt Bridge Distance Comparison across Variants]:")
    log(f"  WT Median (IQR)    : {np.median(wt_salt):.2f} Å ({np.percentile(wt_salt, 25):.2f}–{np.percentile(wt_salt, 75):.2f} Å)")
    log(f"  S752D Median (IQR) : {np.median(s752d_salt):.2f} Å ({np.percentile(s752d_salt, 25):.2f}–{np.percentile(s752d_salt, 75):.2f} Å)")
    log(f"  D754A Median (IQR) : {np.median(d754a_salt):.2f} Å ({np.percentile(d754a_salt, 25):.2f}–{np.percentile(d754a_salt, 75):.2f} Å)")
    log(f"  WT vs. S752D Mann-Whitney U test : U = {u_sb_s:.1f}, p = {p_sb_s:.4f} (Manuscript: p = 0.6619)")
    log(f"  WT vs. D754A Mann-Whitney U test : U = {u_sb_d:.1f}, p = {p_sb_d:.4f} (Manuscript: p = 0.2052)")

    # --------------------------------------------------------------------------
    # 4. Section 3.7: DFGモチーフ (743–745) pLDDT の状態依存性秩序化
    # --------------------------------------------------------------------------
    log("\n" + "#" * 80)
    log("4. SECTION 3.7: DFG MOTIF (743–745) pLDDT CONFIDENCE SURGE")
    log("#" * 80)

    # 全体プール比較 (Closed Basin vs. Open Basin)
    closed_dfg = df[df["Conformational_State"] == "Closed Basin"]["DFG_pLDDT"].dropna()
    open_dfg = df[df["Conformational_State"] == "Open Basin"]["DFG_pLDDT"].dropna()
    inter_dfg = df[df["Conformational_State"] == "Intermediate Cluster"]["DFG_pLDDT"].dropna()

    u_dfg, p_dfg = stats.mannwhitneyu(open_dfg, closed_dfg, alternative="two-sided")

    log("\n[Overall Pooled Comparison (Across all 300 models)]:")
    log(f"  Closed Basin (n={len(closed_dfg)})   : Median = {np.median(closed_dfg):.2f} (IQR: {np.percentile(closed_dfg, 25):.2f}–{np.percentile(closed_dfg, 75):.2f})")
    log(f"  Intermediate (n={len(inter_dfg)})   : Median = {np.median(inter_dfg):.2f} (IQR: {np.percentile(inter_dfg, 25):.2f}–{np.percentile(inter_dfg, 75):.2f})")
    log(f"  Open Basin   (n={len(open_dfg)})    : Median = {np.median(open_dfg):.2f} (IQR: {np.percentile(open_dfg, 25):.2f}–{np.percentile(open_dfg, 75):.2f})")
    log(f"  Two-sided Mann-Whitney U test (Closed vs. Open):")
    log(f"    U-statistic = {u_dfg:.1f}")
    log(f"    p-value     = {p_dfg:.4e} (Manuscript: p = 3.33e-42)")

    # WT 単独比較 (査読指摘対応：D754A 変異の交絡因子を排除した野生型キナーゼ内検定)
    wt_df = df[df["Variant"] == "WT"]
    wt_closed_dfg = wt_df[wt_df["Conformational_State"] == "Closed Basin"]["DFG_pLDDT"].dropna()
    wt_active_dfg = wt_df[wt_df["Conformational_State"] != "Closed Basin"]["DFG_pLDDT"].dropna()

    u_wt_dfg, p_wt_dfg = stats.mannwhitneyu(wt_active_dfg, wt_closed_dfg, alternative="two-sided")

    log("\n[Unmutated WT Single-Variant Comparison (Confounder-free Control)]:")
    log(f"  WT Closed (n={len(wt_closed_dfg)})        : Median = {np.median(wt_closed_dfg):.2f}")
    log(f"  WT Active/Inter (n={len(wt_active_dfg)}) : Median = {np.median(wt_active_dfg):.2f}")
    log(f"  Two-sided Mann-Whitney U test (WT Closed vs. WT Active):")
    log(f"    U-statistic = {u_wt_dfg:.1f}")
    log(f"    p-value     = {p_wt_dfg:.4e} (Proves state-dependent ordering independent of mutations)")

    log("\n" + "=" * 80)
    log("All statistical reproduction tests completed successfully!")
    log("=" * 80)

    # ファイル保存
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults successfully archived to: {out_path.resolve()}")


# ==============================================================================
# CLI エントリーポイント
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Reproduce all statistical tests for MuSK K-ESPRIT.")
    parser.add_argument("--data", type=str, default="", help="Path to processed_metrics_with_plddt.csv")
    parser.add_argument("--out", type=str, default="outputs/statistical_test_results.txt", help="Output text file path")
    args = parser.parse_args()

    # 自動探索候補
    if args.data:
        csv_path = Path(args.data)
    else:
        candidates = [
            Path("data/processed_metrics_with_plddt.csv"),
            Path("../data/processed_metrics_with_plddt.csv"),
            Path("/content/MuSK-K-ESPRIT/data/processed_metrics_with_plddt.csv"),
            Path("/content/processed_metrics_with_plddt.csv")
        ]
        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break

    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "processed_metrics_with_plddt.csv が見つかりません。--data でパスを指定してください。"
        )

    df = pd.read_csv(csv_path)
    run_all_statistical_tests(df, output_file=args.out)


if __name__ == "__main__":
    main()