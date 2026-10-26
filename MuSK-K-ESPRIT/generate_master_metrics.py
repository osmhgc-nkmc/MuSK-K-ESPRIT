# 必要なライブラリのインストール
!pip install -q biopython pandas scipy

import os
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.vectors import calc_dihedral

# ==========================================
# 1. 残基番号の定義（+529 全長番号オフセット）
# ==========================================
OFFSET = 529

# 構造コアおよび各機能部位の残基定義
RES_CORE_E  = list(range(696, 719 + 1))  # C-lobe alphaE-helix (696-719)
RES_CORE_F  = list(range(770, 795 + 1))  # C-lobe alphaF-helix (770-795)
RES_CORE    = RES_CORE_E + RES_CORE_F    # 剛体重ね合わせ用コア (48-50残基)

RES_ALOOP   = list(range(743, 765 + 1))  # Activation loop (743-765, 23残基)
RES_DFG     = [743, 744, 745]            # DFG motif (Asp743, Phe744, Gly745)
RES_PLOOP   = list(range(577, 585 + 1))  # P-loop (577-585)
RES_ALPHAC  = list(range(621, 636 + 1))  # alphaC-helix (621-636)
RES_NLOBE   = list(range(570, 604 + 1))  # N-lobe rigid core (570-604)

# 個別キー残基
RES_LYS609  = 609  # beta3 Lys
RES_GLU626  = 626  # alphaC Glu
RES_MET630  = 630  # RS3
RES_LEU641  = 641  # RS4
RES_HIS723  = 723  # RS1
RES_ALA742  = 742  # X-DFG
RES_ASP743  = 743  # DFG-Asp
RES_PHE744  = 744  # DFG-Phe / RS2
RES_GLY745  = 745  # DFG-Gly
RES_754     = 754  # Asp754 / Ala754 (Latch)
RES_GLN801  = 801  # Gln801 (Latch partner)

TARGET_DIRS = {
    "WT": "/content/WT",
    "S752D": "/content/S752D",
    "D754A": "/content/D754A"
}

# ==========================================
# 2. 幾何・二面角・距離計算補助関数
# ==========================================
def vec(res, atom_name):
    return res[atom_name].get_vector()

def get_dihedral(v1, v2, v3, v4):
    return np.degrees(calc_dihedral(v1, v2, v3, v4))

def get_ramachandran_region(phi, psi):
    if np.isnan(phi) or np.isnan(psi):
        return "-"
    if phi < 0:
        return "A" if -100.0 <= psi <= 50.0 else "B"
    return "L"

def get_chi1_rotamer(chi1):
    if np.isnan(chi1):
        return "-"
    if -120.0 <= chi1 < 0.0:
        return "minus"
    elif 0.0 <= chi1 <= 120.0:
        return "plus"
    return "trans"

def min_heavy_atom_dist(res1, res2):
    """2残基間の非水素（重原子）最小間距離（Å）"""
    if res1 is None or res2 is None:
        return np.nan
    atoms1 = [a for a in res1 if a.element != 'H']
    atoms2 = [a for a in res2 if a.element != 'H']
    if not atoms1 or not atoms2:
        return np.nan
    coords1 = np.array([a.coord for a in atoms1])
    coords2 = np.array([a.coord for a in atoms2])
    dists = np.linalg.norm(coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :], axis=-1)
    return float(np.min(dists))

# ==========================================
# 3. 構造情報のパースとリナンバリング
# ==========================================
def extract_chain_data(structure):
    """構造から全長番号付きの残基辞書を取得"""
    model = structure[0]
    chain = list(model.get_chains())[0]
    res_dict = {}
    for res in chain:
        if res.id[0] == ' ':  # 標準アミノ酸
            full_seq = res.id[1] + OFFSET
            res_dict[full_seq] = res
    return res_dict

# ==========================================
# 4. 全モデルの事前読み込みと WT #27 の特定
# ==========================================
parser = MMCIFParser(QUIET=True)
model_entries = []

print("全300モデルのCIFファイルを探索中...")
for variant, dir_path in TARGET_DIRS.items():
    p = Path(dir_path)
    if not p.exists():
        continue
    files = sorted([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in [".cif", ".cifs", ".mmcif"]])
    print(f"  [{variant}]: {len(files)} ファイル検出")
    for fp in files:
        model_entries.append({
            "variant": variant,
            "filepath": fp,
            "filename": fp.name,
            "id": fp.stem
        })

if len(model_entries) != 300:
    print(f"注意: 合計ファイル数が300件ではなく {len(model_entries)} 件です。")

# 基底メドイド (WT Model #27) のファイルを特定
ref_entry = None
for entry in model_entries:
    if entry["variant"] == "WT" and ("_27_" in entry["filename"] or entry["filename"].endswith("_27.cif") or "model_27" in entry["filename"]):
        ref_entry = entry
        break

if ref_entry is None:
    # 候補が厳密一致しない場合、"27" を含むWTファイルを検索
    candidates = [e for e in model_entries if e["variant"] == "WT" and "27" in e["filename"]]
    if candidates:
        ref_entry = candidates[0]

if ref_entry is None:
    raise FileNotFoundError("WT Model #27 のファイルが見つかりません。ファイル名を確認してください。")

print(f"\n基底参照メドイド (WT Model #27) を設定: {ref_entry['filename']}")
ref_structure = parser.get_structure(ref_entry["id"], str(ref_entry["filepath"]))
ref_res_dict = extract_chain_data(ref_structure)

# WT #27 の Core Cα および A-loop Cα の参照座標・原子リスト
ref_core_ca = [ref_res_dict[r]['CA'] for r in RES_CORE if r in ref_res_dict and 'CA' in ref_res_dict[r]]
ref_aloop_ca = [ref_res_dict[r]['CA'] for r in RES_ALOOP if r in ref_res_dict and 'CA' in ref_res_dict[r]]
ref_core_res_ids = [r for r in RES_CORE if r in ref_res_dict and 'CA' in ref_res_dict[r]]
ref_aloop_res_ids = [r for r in RES_ALOOP if r in ref_res_dict and 'CA' in ref_res_dict[r]]

# ==========================================
# 5. 各モデルのメトリクス計算と統合
# ==========================================
superimposer = Superimposer()
master_records = []

print("\n全300モデルの構造指標、二面角、pLDDT、重畳RMSDを一括計算中...")

for idx, entry in enumerate(model_entries):
    struct = parser.get_structure(entry["id"], str(entry["filepath"]))
    res_dict = extract_chain_data(struct)

    # --- A. 剛体重ね合わせ (Core Cα) と A-loop Cα-RMSD ---
    mob_core_ca = [res_dict[r]['CA'] for r in ref_core_res_ids if r in res_dict and 'CA' in res_dict[r]]
    matched_ref_core_ca = [ref_res_dict[r]['CA'] for r in ref_core_res_ids if r in res_dict and 'CA' in res_dict[r]]

    superimposer.set_atoms(matched_ref_core_ca, mob_core_ca)
    superimposer.apply(struct[0].get_atoms())  # メモリ上で座標を重ね合わせ変換

    # 重畳後の A-loop Cα 座標から RMSD を算出
    matched_ref_aloop_coords = []
    matched_mob_aloop_coords = []
    for r in ref_aloop_res_ids:
        if r in res_dict and 'CA' in res_dict[r]:
            matched_ref_aloop_coords.append(ref_res_dict[r]['CA'].coord)
            matched_mob_aloop_coords.append(res_dict[r]['CA'].coord)

    diff = np.array(matched_mob_aloop_coords) - np.array(matched_ref_aloop_coords)
    aloop_rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))

    # --- B. R-spine 最小重原子間距離 ---
    d_rs3_rs4 = min_heavy_atom_dist(res_dict.get(RES_MET630), res_dict.get(RES_LEU641))
    d_rs2_rs3 = min_heavy_atom_dist(res_dict.get(RES_PHE744), res_dict.get(RES_MET630))
    d_rs1_rs2 = min_heavy_atom_dist(res_dict.get(RES_HIS723), res_dict.get(RES_PHE744))

    # --- C. Latch, 塩橋, クレフト幅 ---
    d_latch = min_heavy_atom_dist(res_dict.get(RES_754), res_dict.get(RES_GLN801))

    # K609(NZ) - E626(OE1/OE2)
    d_salt_bridge = np.nan
    if RES_LYS609 in res_dict and RES_GLU626 in res_dict:
        k_res = res_dict[RES_LYS609]
        e_res = res_dict[RES_GLU626]
        if 'NZ' in k_res:
            oe_atoms = [e_res[a] for a in ['OE1', 'OE2'] if a in e_res]
            if oe_atoms:
                d_salt_bridge = float(min(np.linalg.norm(k_res['NZ'].coord - a.coord) for a in oe_atoms))

    # Cleft Breathing (N-lobe 570-604 Cα重心 - C-lobe core Cα重心)
    nlobe_cas = [res_dict[r]['CA'].coord for r in RES_NLOBE if r in res_dict and 'CA' in res_dict[r]]
    clobe_cas = [res_dict[r]['CA'].coord for r in RES_CORE if r in res_dict and 'CA' in res_dict[r]]
    d_cleft = float(np.linalg.norm(np.mean(nlobe_cas, axis=0) - np.mean(clobe_cas, axis=0))) if nlobe_cas and clobe_cas else np.nan

    # --- D. 各領域の pLDDT ---
    def calc_mean_plddt(res_list):
        scores = []
        for r in res_list:
            if r in res_dict:
                for a in res_dict[r]:
                    scores.append(a.get_bfactor())
        return float(np.mean(scores)) if scores else np.nan

    core_plddt   = calc_mean_plddt(RES_CORE)
    aloop_plddt  = calc_mean_plddt(RES_ALOOP)
    dfg_plddt    = calc_mean_plddt(RES_DFG)
    ploop_plddt  = calc_mean_plddt(RES_PLOOP)
    alphac_plddt = calc_mean_plddt(RES_ALPHAC)

    # --- E. 二面角 (X-DFG) & KinCore 空間距離 ---
    phi_X = get_dihedral(vec(res_dict[RES_ALA742 - 1], 'C'), vec(res_dict[RES_ALA742], 'N'),
                         vec(res_dict[RES_ALA742], 'CA'), vec(res_dict[RES_ALA742], 'C')) if (RES_ALA742 - 1) in res_dict else np.nan
    psi_X = get_dihedral(vec(res_dict[RES_ALA742], 'N'), vec(res_dict[RES_ALA742], 'CA'),
                         vec(res_dict[RES_ALA742], 'C'), vec(res_dict[RES_ASP743], 'N'))

    phi_D = get_dihedral(vec(res_dict[RES_ALA742], 'C'), vec(res_dict[RES_ASP743], 'N'),
                         vec(res_dict[RES_ASP743], 'CA'), vec(res_dict[RES_ASP743], 'C'))
    psi_D = get_dihedral(vec(res_dict[RES_ASP743], 'N'), vec(res_dict[RES_ASP743], 'CA'),
                         vec(res_dict[RES_ASP743], 'C'), vec(res_dict[RES_PHE744], 'N'))

    phi_F = get_dihedral(vec(res_dict[RES_ASP743], 'C'), vec(res_dict[RES_PHE744], 'N'),
                         vec(res_dict[RES_PHE744], 'CA'), vec(res_dict[RES_PHE744], 'C'))
    psi_F = get_dihedral(vec(res_dict[RES_PHE744], 'N'), vec(res_dict[RES_PHE744], 'CA'),
                         vec(res_dict[RES_PHE744], 'C'), vec(res_dict[RES_GLY745], 'N'))

    chi1_F = get_dihedral(vec(res_dict[RES_PHE744], 'N'), vec(res_dict[RES_PHE744], 'CA'),
                          vec(res_dict[RES_PHE744], 'CB'), vec(res_dict[RES_PHE744], 'CG'))

    d1 = float(np.linalg.norm(res_dict[RES_LYS609]['CA'].coord - res_dict[RES_PHE744]['CZ'].coord))
    d2 = float(np.linalg.norm(res_dict[RES_GLU626]['CA'].coord - res_dict[RES_PHE744]['CZ'].coord))

    reg_X = get_ramachandran_region(phi_X, psi_X)
    reg_D = get_ramachandran_region(phi_D, psi_D)
    reg_F = get_ramachandran_region(phi_F, psi_F)
    x_dfg_motif = f"{reg_X}{reg_D}{reg_F}"
    rotamer_F = get_chi1_rotamer(chi1_F)
    dfg_spatial = "DFG-in" if (d1 < 14.0 and d2 < 11.0) else "DFG-out"
    kincore_class = f"{dfg_spatial}/{x_dfg_motif}{rotamer_F}"

    # --- F. 機能的配座状態 (Section 2.7 定義) ---
    if aloop_rmsd < 5.0 and d_rs2_rs3 > 4.5:
        state_class = "Closed Basin"
    elif aloop_rmsd < 10.0 and d_rs2_rs3 <= 4.5:
        state_class = "Intermediate Cluster"
    elif aloop_rmsd >= 10.0 and d_rs2_rs3 <= 4.5:
        state_class = "Open Basin"
    else:
        state_class = "Intermediate Cluster"  # 境界遷移域の補正

    master_records.append({
        "Filename": entry["filename"],
        "Structure_ID": entry["id"],
        "Variant": entry["variant"],
        "Conformational_State": state_class,
        "A_loop_RMSD": round(aloop_rmsd, 3),
        "RS2_RS3_dist": round(d_rs2_rs3, 3),
        "RS1_RS2_dist": round(d_rs1_rs2, 3),
        "RS3_RS4_dist": round(d_rs3_rs4, 3),
        "Latch_dist": round(d_latch, 3),
        "Salt_Bridge_dist": round(d_salt_bridge, 3),
        "Cleft_Width": round(d_cleft, 3),
        "Core_pLDDT": round(core_plddt, 2),
        "A_loop_pLDDT": round(aloop_plddt, 2),
        "DFG_pLDDT": round(dfg_plddt, 2),
        "Ploop_pLDDT": round(ploop_plddt, 2),
        "AlphaC_pLDDT": round(alphac_plddt, 2),
        "phi_F": round(phi_F, 2),
        "psi_F": round(psi_F, 2),
        "chi1_F": round(chi1_F, 2),
        "phi_X": round(phi_X, 2),
        "psi_X": round(psi_X, 2),
        "phi_D": round(phi_D, 2),
        "psi_D": round(psi_D, 2),
        "D1": round(d1, 2),
        "D2": round(d2, 2),
        "X_DFG_Motif": x_dfg_motif,
        "KinCore_Class": kincore_class
    })

# ==========================================
# 6. CSV保存と集計サマリーの表示
# ==========================================
master_df = pd.DataFrame(master_records)
output_csv_path = "/content/processed_metrics_with_plddt.csv"
master_df.to_csv(output_csv_path, index=False)

print("\n" + "=" * 60)
print(f"マスターCSV出力完了: {output_csv_path} (計 {len(master_df)} 行)")
print("=" * 60)

print("\n【配座状態（Conformational_State）の内訳】")
display(pd.crosstab(master_df["Variant"], master_df["Conformational_State"], margins=True))

print("\n【主要メドイドモデルの検証（WT #27, #45, #37）】")
medoid_subset = master_df[master_df["Structure_ID"].str.contains("27|45|37") & (master_df["Variant"] == "WT")]
display(medoid_subset[["Structure_ID", "Conformational_State", "A_loop_RMSD", "RS2_RS3_dist", "Latch_dist", "phi_F", "psi_F", "KinCore_Class"]])