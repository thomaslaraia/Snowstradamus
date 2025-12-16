from scripts.imports import *
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from textwrap import fill

import argparse
import sys
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.patches as mpatches

# -----------------------------------------------------------
# ARGUMENTS
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-E", type=int, default=80)
args = parser.parse_args()

E = args.E
suffix = 'nw_DW_nolof_test'
BIN_W_PARAM = 0
remove_cams = []
num_cameras = 18 - len(remove_cams)

# -----------------------------------------------------------
# OUTPUT FOLDERS (images + logs in same subfolder)
# -----------------------------------------------------------
base_img_dir = "./bootstrap_images"
run_dir = os.path.join(base_img_dir, f"{E}m_{suffix}")
os.makedirs(run_dir, exist_ok=True)

# ------------------------------
# LOGGING: tee stdout to a file
# ------------------------------
log_path = os.path.join(run_dir, f"{E}m_bootstrap_{suffix}.txt")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "w")
orig_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# -----------------------------------------------------------
# LOAD AND PREP DATA
# -----------------------------------------------------------
df = pd.read_pickle(f'dataset_lcforest_noLOF_bin15_th3_{E}m_1kmsmallbox_noprior_ta_dw1_v7.pkl')

df['Eg_strong'] = np.where((df['beam_str'] == 'strong')&(df['outlier'] == 1), df['Eg'], np.nan)
df['Ev_strong'] = np.where((df['beam_str'] == 'strong')&(df['outlier'] == 1), df['Ev'], np.nan)
df['Eg_weak'] = np.where((df['beam_str'] == 'weak')&(df['outlier'] == 1), df['Eg'], np.nan)
df['Ev_weak'] = np.where((df['beam_str'] == 'weak')&(df['outlier'] == 1), df['Ev'], np.nan)
df['forest_type'] = np.where(df['segment_landcover']<119, 'closed', 'open')

df_grouped = df.groupby(['camera','date','lat','lon']).agg({
    'pvpg': 'mean',
    'pv': 'max',
    'pg': 'max',
    'Eg_strong': 'median',
    'Ev_strong': 'median',
    'Eg_weak': 'median',
    'Ev_weak': 'median',
    'data_quantity': 'max',
    'snr': 'mean',
    'FSC': 'mean',
    'TreeSnow': 'mean',
    'layer_flag': 'mean',
    'file_index': 'mean',
    'msw_flag': 'mean',
    'pv_ratio_mean': 'mean',
    'pv_ratio_max': 'mean',
    'forest_type': lambda x: pd.Series.mode(x).iloc[0],
}).reset_index()
df_grouped = df_grouped[df_grouped['Eg_strong']>=0]
df_grouped['JointSnow'] = df_grouped['FSC'] + df_grouped['TreeSnow']
df_grouped = df_grouped[~df_grouped["camera"].isin(remove_cams)]

df_grouped['cell_id'] = (
    df_grouped['camera'].astype(str) + '|' +
    df_grouped['date'].astype(str)   + '|' +
    df_grouped['lat'].round(6).astype(str) + '|' +
    df_grouped['lon'].round(6).astype(str)
)


# ===========================================================
# PHASE 2: Bootstrap
# ===========================================================
EG_COL = "Eg_strong"
EV_COL = "Ev_strong"
Y_BIN_COL  = "JointSnowBinary"
FRAC_W = 1.0              # weight for fractional 0<y<1 in RMSE
N_BOOT = 5
N_SPLITS_CV = 5
RATIO_GRID = np.round(np.arange(1.05, 1.30 + 1e-9, 0.01), 2)  # 1.05..1.30
DQ_GRID    = np.arange(12, 36)                                 # 12..35
TOL_NEAR   = 0.003
RNG = np.random.RandomState(42)

# ------------------------------ helper base conditions ------------------------------
def base_conditions_opt(df):
    return (
        ((df['FSC'] <= 0.005) | (df['FSC'] >= 0.995)) &
        ((df['TreeSnow'] == 0) | (df['TreeSnow'] == 1)) &
        (df['Eg_strong'] >= 0)
    )

def base_conditions_boot(df):
    return ((df['FSC'] >= 0.0) & (df['FSC'] <= 1.0) & (df['Eg_strong'] >= 0))

def apply_filters_for_search(df, ratio_thresh, dq_thresh):
    cond = base_conditions_opt(df) & ((df['Eg_strong']/df['Eg_weak']) >= ratio_thresh) & (df['data_quantity'] >= dq_thresh)
    out = df.loc[cond].copy()
    out['JointSnow'] = out['FSC'] + out['TreeSnow']
    out['JointSnowRounded'] = np.round(out['JointSnow']).astype(int)
    return out

def apply_filters_for_boot(df, ratio_thresh, dq_thresh):
    cond = base_conditions_boot(df) & ((df['Eg_strong']/df['Eg_weak']) >= ratio_thresh) & (df['data_quantity'] >= dq_thresh)
    out = df.loc[cond].copy()
    out['JointSnow'] = out['FSC'] + out['TreeSnow']
    out['JointSnowBinary'] = out['JointSnow'].apply(lambda x: 1 if x >= 1 else x).astype(float)
    return out

# ------------------------------ CV metrics ------------------------------
def assign_folds_by_camera(df, n_splits=5):
    counts = df['camera'].value_counts()
    cams_sorted = counts.index.tolist()
    cam2fold = {cam: (i % n_splits) for i, cam in enumerate(cams_sorted)}
    return df['camera'].map(cam2fold).to_numpy()

def cv_multinomial_metrics(df, df2, features=('Eg_strong','Ev_strong'), n_splits=5):
    if df.shape[0] == 0 or df['JointSnowRounded'].nunique() < 2:
        return np.nan, None, np.nan, np.nan, None, np.nan

    n_unique_cams = df['camera'].nunique()
    n_splits_eff = max(2, min(n_splits, n_unique_cams))

    grp = assign_folds_by_camera(df, n_splits_eff)
    X = df.loc[:, list(features)].to_numpy()
    y = df['JointSnowRounded'].to_numpy()
    X2 = df2.loc[:, list(features)].to_numpy()
    y2 = df2['JointSnowRounded'].to_numpy()

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not np.any(valid):
        return np.nan, None, np.nan, np.nan, None, np.nan
    X, y, grp = X[valid], y[valid], grp[valid]
    
    valid2 = np.isfinite(X2).all(axis=1) & np.isfinite(y2)
    X2, y2 = X2[valid2], y2[valid2]
    
    if np.unique(y).size < 2:
        return np.nan, None, np.nan, np.nan, None, np.nan

    all_true, all_pred = [], []

    for f in range(n_splits_eff):
        test_mask  = (grp == f)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue

        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask],  y[test_mask]

        if np.unique(ytr).size < 2:
            continue

        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=0
        )
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        all_true.extend(yte.tolist())
        all_pred.extend(yhat.tolist())

    if len(all_true) == 0:
        return np.nan, None, np.nan, np.nan, None, np.nan

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=[0,1,2])

    y_true_bin = (all_true >= 1).astype(int)
    y_pred_bin = (all_pred >= 1).astype(int)
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=0)
    model.fit(X, y)
    yhat2 = model.predict(X2)
    oob_acc = accuracy_score(y2, yhat2)
    cm2 = confusion_matrix(y2, yhat2, labels=[0,1,2])
    y_true_bin2 = (y2 >= 1).astype(int)
    y_pred_bin2 = (yhat2 >= 1).astype(int)
    oob_bin_acc = accuracy_score(y_true_bin2, y_pred_bin2)

    return acc, cm, bin_acc, oob_acc, cm2, oob_bin_acc

def grid_search_dedup(dedup_train, dedup_test, ratio_grid, dq_grid):
    rows = []
    for r in ratio_grid:
        for dq in dq_grid:
            df_f = apply_filters_for_search(dedup_train, r, dq)
            df_f2 = apply_filters_for_search(dedup_test, r, dq)
            acc, cm, bin_acc, oob_acc, cm2, oob_bin_acc = cv_multinomial_metrics(df_f, df_f2)
            rows.append({
                'ratio': r,
                'dq': int(dq),
                'accuracy': acc,
                'bin_acc': bin_acc,
                'oob_acc': oob_acc,
                'oob_bin_acc': oob_bin_acc,
                'n_rows': int(len(df_f)),
                'n_rows_test': int(len(df_f2)),
                'conf_mat': cm,
                'conf_mat2': cm2
            })
    res = pd.DataFrame(rows).dropna(subset=['accuracy']).reset_index(drop=True)
    return res

def choose_best(res, tol=0.002):
    if res.empty:
        return None, pd.DataFrame(columns=res.columns)
    best = res['accuracy'].max()
    near = res[res['accuracy'] >= best - tol].copy()
    near = near.sort_values(['n_rows','accuracy','ratio','dq'],
                            ascending=[False, False, True, True]).reset_index(drop=True)
    return near.iloc[0].to_dict(), near

# ------------------------------ angular model ------------------------------
def mod2pi(x): return np.mod(x, 2*np.pi)
def dccw(a, b): return mod2pi(b - a)

def angular_sector_map(eg, ev, cx, cy, theta1, theta2, eps=1e-9):
    eg = np.asarray(eg); ev = np.asarray(ev)
    theta = mod2pi(np.arctan2(ev - cy, eg - cx))
    t1 = mod2pi(theta1); t2 = mod2pi(theta2); pi_m = mod2pi(np.pi)
    arc = dccw(t1, t2); arc = np.maximum(arc, eps)
    d1 = dccw(t1, theta)
    in_grad = d1 <= arc + eps
    vals = np.empty_like(theta, dtype=float)
    vals[in_grad] = np.clip(d1[in_grad] / arc, 0.0, 1.0)
    d_from_t2 = dccw(t2, theta)
    d_t2_to_pi = dccw(t2, pi_m)
    in_high = (~in_grad) & (d_from_t2 < d_t2_to_pi - eps)
    vals[in_high] = 1.0
    vals[~(in_grad | in_high)] = 0.0
    at_pi = np.isclose(mod2pi(theta), pi_m, atol=1e-12); vals[at_pi] = 0.0
    at_center = np.isclose(eg, cx, atol=1e-12) & np.isclose(ev, cy, atol=1e-12); vals[at_center] = 0.0
    return vals

def tiny_arc_penalty(theta1, theta2, thresh=1e-3):
    arc = dccw(mod2pi(theta1), mod2pi(theta2))
    return 1e6 * (thresh - arc + 1e-9) if arc < thresh else 0.0

def weighted_rmse(y_true, y_pred, frac_weight=1.0, bin_weight=0.25):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    w = np.where((y_true > 0) & (y_true < 1), frac_weight, bin_weight)
    return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

def fit_sector_model_with_group_binw(train_df):
    data = train_df.dropna(subset=[EG_COL, EV_COL, Y_BIN_COL]).copy()
    eg = data[EG_COL].values; ev = data[EV_COL].values; y = data[Y_BIN_COL].astype(float).values

    n_frac_total = int(((y > 0) & (y < 1)).sum())
    n_bin_total  = int(len(y) - n_frac_total)
    
    if BIN_W_PARAM == 0:
        BIN_W_GROUP = 1
    else:
        BIN_W_GROUP  = BIN_W_PARAM*(n_frac_total / n_bin_total) if n_bin_total > 0 and n_frac_total > 0 else 1.0

    def init_params():
        return np.array([0.0, 1.8, -np.pi/4, -np.pi/8], dtype=float)

    bounds = [(-2, 0.0), (max(1e-6, 0.0), np.inf), (-np.pi/2, np.pi), (-np.pi, 0.0)]

    def objective(p):
        cx, cy, t1, t2 = p
        y_hat = angular_sector_map(eg, ev, cx, cy, t1, t2)
        return weighted_rmse(y, y_hat, frac_weight=FRAC_W, bin_weight=BIN_W_GROUP) + tiny_arc_penalty(t1, t2)

    res = minimize(objective, init_params(), method="L-BFGS-B", bounds=bounds)
    cx, cy, t1, t2 = res.x
    params = {"cx": cx, "cy": cy, "theta1": t1, "theta2": t2, "BIN_W_GROUP": BIN_W_GROUP}
    return params

def predict_sector(df, params):
    eg = df[EG_COL].values; ev = df[EV_COL].values
    p  = angular_sector_map(eg, ev, params["cx"], params["cy"], params["theta1"], params["theta2"])
    return np.clip(p, 0.0, 1.0)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    overall_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    overall_bias = float(np.mean(y_pred - y_true))
    frac_mask = (y_true > 0) & (y_true < 1)
    frac_rmse = float(np.sqrt(mean_squared_error(y_true[frac_mask], y_pred[frac_mask]))) if np.any(frac_mask) else np.nan
    frac_bias = float(np.mean(y_pred[frac_mask] - y_true[frac_mask])) if np.any(frac_mask) else np.nan
    none_mask = (y_true == 0)
    none_rmse = float(np.sqrt(mean_squared_error(y_true[none_mask], y_pred[none_mask]))) if np.any(none_mask) else np.nan
    none_bias = float(np.mean(y_pred[none_mask] - y_true[none_mask])) if np.any(none_mask) else np.nan
    full_mask = (y_true == 1)
    full_rmse = float(np.sqrt(mean_squared_error(y_true[full_mask], y_pred[full_mask]))) if np.any(full_mask) else np.nan
    full_bias = float(np.mean(y_pred[full_mask] - y_true[full_mask])) if np.any(full_mask) else np.nan
    return dict(
        overall_rmse=overall_rmse, overall_bias=overall_bias,
        overall_frac_rmse=frac_rmse, overall_frac_bias=frac_bias,
        overall_none_rmse=none_rmse, overall_none_bias=none_bias,
        overall_full_rmse=full_rmse, overall_full_bias=full_bias
    )

# ------------------------------ bootstrap loop ------------------------------
all_cameras = sorted(df_grouped['camera'].unique())
assert len(all_cameras) == num_cameras, f"Expected {num_cameras} unique cameras, found {len(all_cameras)}."

phase2_rows = []
all_oob_y_true = []
all_oob_y_pred = []
all_oob_cams   = []

cumulative_oob_conf_mat = np.zeros((3,3), dtype=int)

sample_oob_df = None
sample_params = None

test_counts = []
test_counts_0 = []
test_counts_1 = []
test_counts_p = []

unique_oob_cells = {}  # key: cell_id -> {0,1,2}

# NEW: per-bootstrap per-camera metrics for mean/median later
per_cam_boot_rows = []

for b in range(N_BOOT):
    start = time.time()
    
    sampled_cams = RNG.choice(all_cameras, size=len(all_cameras), replace=True)
    sampled_unique = sorted(set(sampled_cams))
    oob_cams = sorted(set(all_cameras) - set(sampled_unique))

    boot_concat = pd.concat(
        [df_grouped[df_grouped['camera'] == cam] for cam in sampled_cams],
        ignore_index=True
    )

    dedup_train = boot_concat.copy()
    dedup_test = df_grouped[df_grouped['camera'].isin(oob_cams)].copy()

    res = grid_search_dedup(dedup_train, dedup_test, RATIO_GRID, DQ_GRID)
    chosen, near = choose_best(res, tol=TOL_NEAR)

    print(f"\n=== Bootstrap {b+1}/{N_BOOT} ===")
    if chosen is None:
        print("No valid filter produced a trainable dataset for search.")
        phase2_rows.append({
            'bootstrap': b+1, 'ratio': np.nan, 'dq': np.nan,
            'n_rows_search': 0, 'n_rows_boot_train': 0, 'n_rows_oob': 0,
            'oob_rmse': np.nan, 'oob_bias': np.nan, 'oob_frac_rmse': np.nan, 'oob_frac_bias': np.nan,
            'oob_none_rmse': np.nan, 'oob_none_bias': np.nan, 'oob_full_rmse': np.nan, 'oob_full_bias': np.nan,
            'n_oob_cameras': len(oob_cams),
            'cv_acc': np.nan,
            'cv_bin_acc': np.nan,
            'oob_acc': np.nan, 'oob_bin_acc': np.nan
        })
        continue

    print(
        f"\nChosen filter -> Eg_strong/Eg_weak >= {chosen['ratio']:.2f}, data_quantity >= {int(chosen['dq'])} "
        f"| CV acc={chosen['accuracy']:.4f}, CV bin acc={chosen['bin_acc']:.4f}, n_rows(dedup)={int(chosen['n_rows'])}"
        f"| OOB acc={chosen['oob_acc']:.4f}, OOB bin acc={chosen['oob_bin_acc']:.4f}, "
        f"n_rows_test(dedup)={int(chosen['n_rows_test'])} |"
    )

    if isinstance(chosen.get('conf_mat2', None), np.ndarray):
        cumulative_oob_conf_mat += chosen['conf_mat2'].astype(int)

    boot_train = apply_filters_for_boot(boot_concat, chosen['ratio'], int(chosen['dq']))

    if len(boot_train) == 0:
        print("Bootstrapped training set empty after filters; skipping.")
        phase2_rows.append({
            'bootstrap': b+1, 'ratio': float(chosen['ratio']), 'dq': int(chosen['dq']),
            'n_rows_search': int(chosen['n_rows']), 'n_rows_boot_train': 0, 'n_rows_oob': int(chosen['n_rows_test']),
            'oob_rmse': np.nan, 'oob_bias': np.nan, 'oob_frac_rmse': np.nan, 'oob_frac_bias': np.nan,
            'oob_none_rmse': np.nan, 'oob_none_bias': np.nan, 'oob_full_rmse': np.nan, 'oob_full_bias': np.nan,
            'n_oob_cameras': len(oob_cams),
            'cv_acc': float(chosen['accuracy']),
            'cv_bin_acc': float(chosen['bin_acc']),
            'oob_acc': float(chosen['oob_acc']),
            'oob_bin_acc': float(chosen['oob_bin_acc'])
        })
        continue

    params = fit_sector_model_with_group_binw(boot_train)

    oob_df = df_grouped[df_grouped['camera'].isin(oob_cams)].copy()
    oob_df = apply_filters_for_boot(oob_df, chosen['ratio'], int(chosen['dq']))
    
    if not oob_df.empty:
        vals = oob_df[Y_BIN_COL].values
        ids  = oob_df['cell_id'].values
        cls = np.where(vals == 0, 0, np.where(vals == 1, 1, 2))
        for cid, c in zip(ids, cls):
            if cid not in unique_oob_cells:
                unique_oob_cells[cid] = int(c)
    
    test_counts.append(len(oob_df))
    test_counts_0.append(len(oob_df[oob_df[Y_BIN_COL]==0]))
    test_counts_1.append(len(oob_df[oob_df[Y_BIN_COL]==1]))
    test_counts_p.append(len(oob_df[(oob_df[Y_BIN_COL]>0)&(oob_df[Y_BIN_COL]<1)]))

    if len(oob_df) > 0:
        y_pred = predict_sector(oob_df, params)
        y_true = oob_df[Y_BIN_COL].astype(float).values
        cams   = oob_df['camera'].values

        # store all predictions
        all_oob_y_true.append(y_true.copy())
        all_oob_y_pred.append(y_pred.copy())
        all_oob_cams.append(cams.copy())

        if sample_oob_df is None:
            sample_oob_df = oob_df.copy()
            sample_params = params
            sample_y_true = y_true.copy()
            sample_y_pred = y_pred.copy()
        
        m = compute_metrics(y_true, y_pred)
        print(f"OOB cameras: {oob_cams if oob_cams else 'none (all cameras sampled)'}")
        print(
            f"OOB n={len(oob_df)} | RMSE={m['overall_rmse']:.4f} | Bias={m['overall_bias']:.4f} | "
            f"FracRMSE={m['overall_frac_rmse'] if np.isfinite(m['overall_frac_rmse']) else np.nan:.4f} | "
            f"FracBias={m['overall_frac_bias'] if np.isfinite(m['overall_frac_bias']) else np.nan:.4f} | "
            f"NoneBias={m['overall_none_bias'] if np.isfinite(m['overall_none_bias']) else np.nan:.4f} | "
            f"FullBias={m['overall_full_bias'] if np.isfinite(m['overall_full_bias']) else np.nan:.4f}"
        )

        # NEW: per-camera metrics for this bootstrap
        unique_cams_b = np.unique(cams)
        for cam_name in unique_cams_b:
            mask_cam = (cams == cam_name)
            if not np.any(mask_cam):
                continue
            mc = compute_metrics(y_true[mask_cam], y_pred[mask_cam])
            row_cam = {"bootstrap": b+1, "camera": cam_name}
            row_cam.update(mc)
            per_cam_boot_rows.append(row_cam)

    else:
        m = dict(overall_rmse=np.nan, overall_bias=np.nan, overall_frac_rmse=np.nan, overall_frac_bias=np.nan,
                 overall_none_rmse=np.nan, overall_none_bias=np.nan, overall_full_rmse=np.nan, overall_full_bias=np.nan)
        print("No OOB rows after filtering (all cameras sampled and/or filtered out).")

    phase2_rows.append({
        'bootstrap': b+1,
        'ratio': float(chosen['ratio']),
        'dq': int(chosen['dq']),
        'n_rows_search': int(chosen['n_rows']),
        'n_rows_boot_train': int(len(boot_train)),
        'n_rows_oob': int(len(oob_df)),
        'oob_rmse': m['overall_rmse'],
        'oob_bias': m['overall_bias'],
        'oob_frac_rmse': m['overall_frac_rmse'],
        'oob_frac_bias': m['overall_frac_bias'],
        'oob_none_rmse': m['overall_none_rmse'],
        'oob_none_bias': m['overall_none_bias'],
        'oob_full_rmse': m['overall_full_rmse'],
        'oob_full_bias': m['overall_full_bias'],
        'n_oob_cameras': len(oob_cams),
        'bin_w_group': params.get('BIN_W_GROUP', np.nan),
        'cv_acc': float(chosen['accuracy']),
        'cv_bin_acc': float(chosen['bin_acc']),
        'oob_acc': float(chosen['oob_acc']),
        'oob_bin_acc': float(chosen['oob_bin_acc'])
    })

    end = time.time()
    print(f"{round(end-start,2)}s")

# ------------------------------ summary ------------------------------
phase2_df = pd.DataFrame(phase2_rows)

print("\n====================")
print("\nFilter choice frequency across bootstraps:")
freq = (phase2_df
        .dropna(subset=['ratio','dq'])
        .value_counts(subset=['ratio','dq'])
        .reset_index(name='count')
        .sort_values('count', ascending=False))
print(freq.to_string(index=False))

print("\nOOB metrics (mean ± std, variance) across bootstraps (ignoring NaNs):")

def summarize_series(name, series):
    mean = np.nanmean(series)
    std = np.nanstd(series)
    var = np.nanvar(series)
    print(f"{name:<12} {mean:.4f} ± {std:.4f} (var={var:.4f})")

summarize_series("RMSE:",       phase2_df['oob_rmse'])
summarize_series("Bias:",       phase2_df['oob_bias'])
summarize_series("Frac RMSE:",  phase2_df['oob_frac_rmse'])
summarize_series("Frac Bias:",  phase2_df['oob_frac_bias'])
summarize_series("0%SC Bias:",  phase2_df['oob_none_bias'])
summarize_series("100%SC Bias:",phase2_df['oob_full_bias'])

print("\nCV metrics (mean± std, variance across chosen filters per bootstrap):")
summarize_series("OOB acc:",      phase2_df['oob_acc'])
summarize_series("OOB bin acc:",  phase2_df['oob_bin_acc'])

print("\nOOB metrics (median across bootstraps, ignoring NaNs):")
print(f"RMSE (median):        {np.nanmedian(phase2_df['oob_rmse']):.4f}")
print(f"Bias (median):        {np.nanmedian(phase2_df['oob_bias']):.4f}")
print(f"Frac RMSE (median):   {np.nanmedian(phase2_df['oob_frac_rmse']):.4f}")
print(f"Frac Bias (median):   {np.nanmedian(phase2_df['oob_frac_bias']):.4f}")
print(f"0%SC Bias (median):   {np.nanmedian(phase2_df['oob_none_bias']):.4f}")
print(f"100%SC Bias (median): {np.nanmedian(phase2_df['oob_full_bias']):.4f}")

print("\nCV metrics (median across bootstraps):")
print(f"OOB acc (median):     {np.nanmedian(phase2_df['oob_acc']):.4f}")
print(f"OOB bin acc (median): {np.nanmedian(phase2_df['oob_bin_acc']):.4f}")

print(f"\nTotal Cells: {np.sum(test_counts)}")
print(f"Total Non-Snow Cells: {np.sum(test_counts_0)}")
print(f"Total Snow Cells: {np.sum(test_counts_1)}")
print(f"Total Partial Snow Cells: {np.sum(test_counts_p)}")

uniq_vals = np.fromiter(unique_oob_cells.values(), dtype=int) if unique_oob_cells else np.array([], dtype=int)
n_unique_total  = uniq_vals.size
n_unique_0      = int((uniq_vals == 0).sum())
n_unique_1      = int((uniq_vals == 1).sum())
n_unique_partial= int((uniq_vals == 2).sum())

print("\nUNIQUE OOB cells across all bootstraps (post-filter):")
print(f"Unique Cells:           {n_unique_total}")
print(f"Unique Non-Snow Cells:  {n_unique_0}")
print(f"Unique Snow Cells:      {n_unique_1}")
print(f"Unique Partial Cells:   {n_unique_partial}")

# --- Cumulative CV confusion matrix (3x3) PLOT ---
cm = cumulative_oob_conf_mat.astype(float)
row_sums = cm.sum(axis=1, keepdims=True)
pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

labels = ["No Snow", "Ground Snow", "Ground + Canopy Snow"]

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = int(cm[i, j])
        perc  = pct[i, j]
        ax.text(
            j, i,
            f"{count}\n({perc:.1f}%)",
            ha="center", va="center",
            fontsize=12,
            color="white" if perc > 50 else "black",
            fontweight="bold" if perc > 50 else "normal"
        )

ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Observed", fontsize=12)
ax.set_title("Cumulative Binary CV Confusion Matrix", fontsize=14, weight="bold")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("% of Observed Class", rotation=90)

ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5, alpha=0.8)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, f"{E}m_confusion_matrix_{suffix}.png"))

# =============================
# GLOBAL FSC ACCURACY BOXPLOTS
# =============================
metrics_labels = ["RMSE", "Bias", "Fractional RMSE", "Fractional Bias", "0%SC Error", "100%SC Error"]
series_list = [
    phase2_df['oob_rmse'],
    phase2_df['oob_bias'],
    phase2_df['oob_frac_rmse'],
    phase2_df['oob_frac_bias'],
    phase2_df['oob_none_bias'],
    phase2_df['oob_full_bias'],
]

# convert to percentages
box_data = [np.array(s, dtype=float) * 100.0 for s in series_list]

# Boxplot showing distribution + mean markers
plt.figure(figsize=(10, 6))
bp = plt.boxplot(
    box_data,
    labels=metrics_labels,
    showmeans=True,
    meanline=False
)
plt.ylabel("Value (%)")
plt.title("ICESat-2 FSC Estimation Metrics - OOB Test Data (Bootstrapped Distributions)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, f"{E}m_FSC_accuracy_boxplots_{suffix}.png"))

# =============================
# PER-CAMERA MEAN & MEDIAN TABLES
# =============================
if per_cam_boot_rows:
    per_cam_boot = pd.DataFrame(per_cam_boot_rows)

    metric_keys = [
        "overall_rmse",
        "overall_bias",
        "overall_frac_rmse",
        "overall_frac_bias",
        "overall_none_bias",
        "overall_full_bias",
    ]

    per_cam_mean_stats = per_cam_boot.groupby("camera")[metric_keys].mean().reset_index()
    per_cam_median_stats = per_cam_boot.groupby("camera")[metric_keys].median().reset_index()

    def build_per_cam_table(df_cam):
        records = []
        for _, row in df_cam.iterrows():
            rec = {"camera": row["camera"]}
            rec["RMSE"]             = row["overall_rmse"] * 100.0 if np.isfinite(row["overall_rmse"]) else np.nan
            rec["Bias"]             = row["overall_bias"] * 100.0 if np.isfinite(row["overall_bias"]) else np.nan
            rec["Fractional RMSE"]  = row["overall_frac_rmse"] * 100.0 if np.isfinite(row["overall_frac_rmse"]) else np.nan
            rec["Fractional Bias"]  = row["overall_frac_bias"] * 100.0 if np.isfinite(row["overall_frac_bias"]) else np.nan
            rec["0%SC Error"]       = row["overall_none_bias"] * 100.0 if np.isfinite(row["overall_none_bias"]) else np.nan
            rec["100%SC Error"]     = row["overall_full_bias"] * 100.0 if np.isfinite(row["overall_full_bias"]) else np.nan
            records.append(rec)
        return pd.DataFrame.from_records(records)

    per_cam_mean_table = build_per_cam_table(per_cam_mean_stats)
    per_cam_median_table = build_per_cam_table(per_cam_median_stats)

    print("\nPer-camera OOB metrics (MEAN across bootstraps, values in %):")
    print(per_cam_mean_table.to_string(index=False))

    print("\nPer-camera OOB metrics (MEDIAN across bootstraps, values in %):")
    print(per_cam_median_table.to_string(index=False))

    # -----------------------------
    # ICESat-2-only per-camera bar (mean)
    # -----------------------------
    metrics_order_pc = ["RMSE", "Bias", "Fractional RMSE", "Fractional Bias", "0%SC Error", "100%SC Error"]
    cams_list = sorted(per_cam_mean_table["camera"].unique())
    xidx = np.arange(len(cams_list))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_order_pc):
        ax = axes[idx]
        vals = (
            per_cam_mean_table
            .set_index("camera")
            .reindex(cams_list)[metric]
            .to_numpy()
        )

        ax.bar(
            xidx,
            vals,
            width=0.8,
            edgecolor="black",
            color="#9e9e9e"
        )

        ax.set_title(metric)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xidx)
        ax.set_xticklabels(cams_list, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if idx in (0, 3):
            ax.set_ylabel("Value (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{E}m_FSC_accuracy_per_camera_ICESat2_mean_{suffix}.png"))

    # =============================
    # ALL-METHODS PER-CAMERA PLOTS (MEAN & MEDIAN)
    # =============================

    optical_txt = """\
bartlett Dozier 23.177501 -9.439586 51.292384 -36.519973 1.324260 -13.676816
bartlett Klein 31.286706 -17.480893 54.492234 -39.486087 0.023964 -33.185616
delta_junction Dozier 20.353743 -6.442907 31.513821 -15.635219 0.394207 -0.238539
delta_junction Klein 26.186551 -11.474010 40.561000 -27.336090 0.002856 -0.339697
glees Dozier 7.905802 0.405437 35.080212 21.023792 0.024953 -0.975714
glees Klein 17.827745 -6.595540 28.748745 14.789280 0.000000 -11.791970
hyytiala Dozier 8.850217 -0.386190 16.079895 -1.267977 0.338278 -0.874679
hyytiala Klein 20.795389 -11.084574 35.006164 -29.096263 0.000000 -11.778477
kenttarova Dozier 1.791885 0.021508 1.519597 1.503822 0.348821 -0.390707
kenttarova Klein 1.800754 -0.109265 1.963569 1.782328 0.000000 -0.481086
lacclair Dozier 10.101989 -1.548306 21.951521 -4.824784 0.225539 -1.411132
lacclair Klein 15.685981 -6.147067 31.636798 -17.752585 0.000000 -5.564035
marcell Dozier 13.988816 6.238930 25.835872 21.492385 0.411054 -0.872326
marcell Klein 12.558183 -2.087138 20.966287 -0.615102 0.000000 -7.630934
old_jack_pine Dozier 6.698874 0.323764 23.113329 3.670988 0.124423 -0.085481
old_jack_pine Klein 7.030053 -1.865420 23.102667 -14.154256 0.000000 -1.496505
oregon Dozier 17.663054 3.770347 24.695605 5.044341 3.289544 -0.027197
oregon Klein 20.451245 -8.334256 32.331926 -20.892467 0.046231 -0.028879
queens Dozier 5.023730 1.330654 NaN NaN 3.009314 -7.062647
queens Klein 13.898831 -5.220205 NaN NaN 0.108507 -31.863760
sodankyla Dozier 18.742372 8.869208 28.982600 20.890652 0.507626 -0.280448
sodankyla Klein 12.331347 4.148149 19.068363 10.077289 0.000000 -0.304331
tammela Dozier 3.233068 0.311490 NaN NaN 1.647362 -1.692317
tammela Klein 21.604560 -10.551700 NaN NaN 0.317490 -26.855485
torgnon Dozier 21.969062 -7.043571 27.948304 -10.037328 3.362859 -10.553833
torgnon Klein 40.447217 -28.717821 51.500637 -43.265937 0.783909 -32.608286
underc Dozier 5.836515 -1.937693 9.547125 -2.884247 0.303866 -4.392615
underc Klein 17.487585 -9.586980 22.911795 -14.990208 0.001686 -19.392549
underhill Dozier 8.118593 -0.628956 12.311025 -0.806946 0.148081 -2.081644
underhill Klein 23.243519 -12.724962 31.365492 -22.016195 0.000000 -21.051755
varrio Dozier 4.944348 0.293982 18.744298 4.419411 0.071175 -0.049292
varrio Klein 2.889224 -0.131979 10.924121 -1.411390 0.000000 -0.052501
willowcreek Dozier 14.118743 -3.691059 42.344031 -32.900268 0.044582 -0.180725
willowcreek Klein 21.774535 -7.864511 64.505335 -63.510410 0.000000 -2.423396
wslcreek Dozier 21.920128 -8.392369 NaN NaN 3.099949 -17.011607
wslcreek Klein 36.803702 -21.816081 NaN NaN 0.000000 -38.178141
"""

    df_opt = pd.read_csv(
        StringIO(optical_txt),
        sep=r"\s+",
        names=[
            "camera",
            "method",
            "RMSE",
            "Bias",
            "Partial Snow RMSE",
            "Partial Snow Bias",
            "0%SC Error",
            "100%SC Error",
        ],
    )

    methods = ["ICESat-2", "Dozier", "Klein"]
    method_colors = {
        "ICESat-2": "#ff7f0e",
        "Dozier":   "#1f77b4",
        "Klein":    "#2ca02c",
    }
    metrics_order_full = [
        "RMSE",
        "Bias",
        "Partial Snow RMSE",
        "Partial Snow Bias",
        "0%SC Error",
        "100%SC Error",
    ]

    def make_df_allmethods(per_cam_table, label_for_title):
        df_ice = per_cam_table.copy()
        df_ice = df_ice.rename(columns={
            "Fractional RMSE": "Partial Snow RMSE",
            "Fractional Bias": "Partial Snow Bias",
        })
        df_ice["method"] = "ICESat-2"
        df_all = pd.concat([df_ice, df_opt], ignore_index=True)
        df_all = df_all.sort_values(["camera", "method"]).reset_index(drop=True)
        return df_all

    def plot_per_camera_allmethods(df_all, file_suffix, fig_title):
        cams_full = sorted(df_all["camera"].unique())
        n_cams = len(cams_full)
        n_methods = len(methods)
        y_idx = np.arange(n_cams)
        bar_height = 0.8 / n_methods

        metric_limits = {}
        for metric in metrics_order_full:
            vals = df_all[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = vals.min(), vals.max()
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
            metric_limits[metric] = (vmin, vmax)

        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        plt.rcParams.update({
            "font.size": 15,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "figure.titlesize": 18,
        })
        axes = axes.ravel()

        nan_frac = 0.075

        for idx, metric in enumerate(metrics_order_full):
            ax = axes[idx]
            vmin, vmax = metric_limits[metric]

            vals = df_all[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            has_pos = np.any(vals > 0)
            has_neg = np.any(vals < 0)

            if has_pos and not has_neg:
                regime = "all_pos"
            elif has_neg and not has_pos:
                regime = "all_neg"
            else:
                regime = "mixed"

            if regime == "all_pos":
                xmin = 0.0
                xmax = vmax * 1.1
                span = xmax - xmin
                nan_length = nan_frac * span
                nan_left = 0.0
            elif regime == "all_neg":
                xmax = 0.0
                xmin = vmin * 1.1
                span = xmax - xmin
                nan_length = nan_frac * span
                nan_left = -nan_length
            else:
                data_min = vmin
                data_max = vmax
                pad = 0.1 * (data_max - data_min if data_max != data_min else 1.0)
                xmin = data_min - pad
                xmax = data_max + pad
                span = xmax - xmin
                nan_length = nan_frac * span
                nan_left = -0.5 * nan_length

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.5, n_cams - 0.5)

            for j, method in enumerate(methods):
                sub = (
                    df_all[df_all["method"] == method]
                    .set_index("camera")
                    .reindex(cams_full)
                )
                vals_m = sub[metric].to_numpy(dtype=float)
                offset = (j - (n_methods - 1) / 2) * bar_height

                for cam_idx, val in enumerate(vals_m):
                    yi = y_idx[cam_idx] + offset

                    if np.isnan(val):
                        ax.barh(
                            yi,
                            nan_length,
                            left=nan_left,
                            height=bar_height,
                            facecolor="white",
                            edgecolor=method_colors[method],
                            hatch="///",
                            linewidth=1.5,
                        )
                    else:
                        ax.barh(
                            yi,
                            val,
                            height=bar_height,
                            color=method_colors.get(method, None),
                            edgecolor="black",
                        )

            ax.set_title(metric)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.grid(axis="x", linestyle="--", alpha=0.4)

            if idx % 2 == 0:
                ax.set_yticks(y_idx)
                ax.set_yticklabels(cams_full, fontsize=15)
                ax.invert_yaxis()
            else:
                ax.set_yticks(y_idx)
                ax.set_yticklabels(cams_full, fontsize=15)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.invert_yaxis()

            if idx in (4, 5):
                ax.set_xlabel("Value (%)", fontsize=15)

        method_patches = [
            mpatches.Patch(color=method_colors[m], label=m)
            for m in methods
        ]

        leg1 = axes[0].legend(
            handles=method_patches,
            title="Method",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )

        nan_patch = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            linewidth=1.5,
            label="No data",
        )

        leg2 = axes[0].legend(
            handles=[nan_patch],
            title="Data availability",
            bbox_to_anchor=(1.02, 0.6),
            loc="upper left",
        )

        axes[0].add_artist(leg1)

        fig.suptitle(fig_title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, file_suffix))

    df_all_mean = make_df_allmethods(per_cam_mean_table, "Mean")
    df_all_median = make_df_allmethods(per_cam_median_table, "Median")

    plot_per_camera_allmethods(
        df_all_mean,
        f"{E}m_FSC_accuracy_per_camera_allmethods_mean_{suffix}.png",
        f"Per-camera OOB metrics (mean across bootstraps, {E}m, {suffix})"
    )
    plot_per_camera_allmethods(
        df_all_median,
        f"{E}m_FSC_accuracy_per_camera_allmethods_median_{suffix}.png",
        f"Per-camera OOB metrics (median across bootstraps, {E}m, {suffix})"
    )

# =============================
# BINARY DISTRIBUTION & SCATTER
# =============================
y_true_all = np.concatenate(all_oob_y_true) if len(all_oob_y_true) else np.array([])
y_pred_all = np.concatenate(all_oob_y_pred) if len(all_oob_y_pred) else np.array([])
cam_all    = np.concatenate(all_oob_cams)   if len(all_oob_cams)   else np.array([])

def plot_binary_prediction_distribution(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask0 = y_true == 0
    mask1 = y_true == 1

    bins = np.arange(0.0, 1.05, 0.05)
    c0, _ = np.histogram(y_pred[mask0], bins=bins)
    c1, _ = np.histogram(y_pred[mask1], bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(7,5))
    plt.plot(centers, c0, label="Snow-Free Ground")
    plt.fill_between(centers, 0, c0, alpha=0.3)
    plt.plot(centers, c1, label="Snow-Covered Ground")
    plt.fill_between(centers, 0, c1, alpha=0.3)
    plt.xlabel("Predicted FSC")
    plt.ylabel("Count")
    plt.title("Distribution of Predictions for Observed Binary Snow Cover - OOB Test Data (All Bootstraps)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{E}m_binary_distribution_{suffix}.png"))

def plot_obs_vs_pred(y_true, y_pred, title="Observed vs Predicted FSC"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.35, edgecolor='k', linewidth=0.3, s=18)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{E}m_observed_predicted_{suffix}.png"))

if y_true_all.size and y_pred_all.size:
    plot_binary_prediction_distribution(y_true_all, y_pred_all)
    plot_obs_vs_pred(y_true_all, y_pred_all, title="Observed vs Predicted FSC - OOB Test Data (All Bootstraps)")

# --- Sample contour plot ---
GRID_N = 300
def plot_test_contour(test_df, params, title="FSC Estimation � OOB Test Data"):
    eg = test_df[EG_COL].values
    ev = test_df[EV_COL].values
    y  = test_df[Y_BIN_COL].astype(float).values

    eg_min, eg_max = float(np.min(eg)), float(np.max(eg))
    ev_min, ev_max = float(np.min(ev)), float(np.max(ev))

    eg_vals = np.linspace(eg_min, eg_max, GRID_N)
    ev_vals = np.linspace(ev_min, ev_max, GRID_N)
    EG, EV = np.meshgrid(eg_vals, ev_vals)

    Z = angular_sector_map(EG, EV, params["cx"], params["cy"], params["theta1"], params["theta2"])

    fig, ax = plt.subplots(figsize=(8,6))
    cs = ax.contourf(EG, EV, Z, levels=np.linspace(0,1,21), cmap='RdBu_r', alpha=0.75)
    cbar = fig.colorbar(cs, ax=ax); cbar.set_label("Predicted Snow Fraction")
    lines = ax.contour(EG, EV, Z, levels=[0,0.25,0.5,0.75,1], colors='k', linestyles='--')
    ax.clabel(lines, fmt='%1.2f')

    sc = ax.scatter(eg, ev, c=y, cmap='RdBu_r', edgecolor='k', s=20, vmin=0, vmax=1, alpha=0.9)
    ax.set_xlabel("Median Ground Radiometry")
    ax.set_ylabel("Median Canopy Radiometry")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{E}m_sample_contour_plot_{suffix}.png"))

if sample_oob_df is not None and sample_params is not None:
    plot_test_contour(sample_oob_df, sample_params, title="FSC Contour Plot with Sample OOB Test Data")

# ------------------------------
# CLOSE LOG FILE / RESTORE STDOUT
# ------------------------------
sys.stdout = orig_stdout
log_file.close()
