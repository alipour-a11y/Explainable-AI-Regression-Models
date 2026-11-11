import os
import glob
import copy
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# =========================================================
# 1. Orthogonal PLS (OPLS)
# =========================================================
class OPLS:
    def __init__(self, n_filter=1, n_pls=4):
        self.n_filter = n_filter
        self.n_pls = n_pls
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        
    def fit(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.y_mean = y.mean()
        E = X - self.X_mean
        f = y - self.y_mean
        self.w_ortho = []
        self.p_ortho = []
        
        for _ in range(self.n_filter):
            w = np.dot(E.T, f) / np.dot(f.T, f)
            w_norm = np.linalg.norm(w)
            if w_norm < 1e-10:
                break
            w /= w_norm
            t = np.dot(E, w)
            p = np.dot(E.T, t) / np.dot(t.T, t)
            
            w_ort = p - w * np.dot(w.T, p)
            norm = np.linalg.norm(w_ort)
            if norm < 1e-10:
                break
            w_ort /= norm
            
            t_ort = np.dot(E, w_ort)
            p_ort = np.dot(E.T, t_ort) / np.dot(t_ort.T, t_ort)
            E = E - np.outer(t_ort, p_ort)
            
            self.w_ortho.append(w_ort)
            self.p_ortho.append(p_ort)
            
        self.pls.fit(E, f)
        return self
        
    def predict(self, X):
        E = X - self.X_mean
        for w_ort, p_ort in zip(self.w_ortho, self.p_ortho):
            t_ort = np.dot(E, w_ort)
            E = E - np.outer(t_ort, p_ort)
        return self.pls.predict(E).ravel() + self.y_mean

# =========================================================
# 2. Interval PLS (iPLS)
# =========================================================
class iPLS:
    def __init__(self, intervals=10, n_pls=5, n_splits=5):
        self.intervals = intervals
        self.n_pls = n_pls
        self.n_splits = n_splits
        self.start_idx = 0
        self.end_idx = 0
        
    def fit(self, X, y):
        n_features = X.shape[1]
        n_intervals = min(self.intervals, n_features)
        step = max(1, n_features // n_intervals)
        best_rmse = float('inf')
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for i in range(n_intervals):
            start = i * step
            end = (i + 1) * step if i < n_intervals - 1 else n_features
            X_sub = X[:, start:end]
            
            y_pred = np.zeros_like(y)
            n_c = min(self.n_pls, X_sub.shape[1], len(y) - (len(y) // self.n_splits) - 1)
            n_c = max(1, n_c)
            
            for tr_idx, val_idx in kf.split(X_sub):
                pls = PLSRegression(n_components=n_c)
                pls.fit(X_sub[tr_idx], y[tr_idx])
                y_pred[val_idx] = pls.predict(X_sub[val_idx]).ravel()
                
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                self.start_idx = start
                self.end_idx = end
                
        n_c_final = min(self.n_pls, self.end_idx - self.start_idx, len(y) - 2)
        n_c_final = max(1, n_c_final)
        self.pls = PLSRegression(n_components=n_c_final)
        self.pls.fit(X[:, self.start_idx:self.end_idx], y)
        return self
        
    def predict(self, X):
        return self.pls.predict(X[:, self.start_idx:self.end_idx]).ravel()

# =========================================================
# 3. Helper Functions
# =========================================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sd = np.std(y_true, ddof=1) if len(y_true) > 1 else np.std(y_true)
    rpd = sd / rmse if rmse > 1e-12 else np.nan
    return r2, rmse, rpd

def fit_and_cross_validate(model_type, X_train, y_train, max_lv=15, n_splits=5):
    n_samples, n_features = X_train.shape
    max_train_samples = n_samples - (n_samples // n_splits)
    
    upper_lv = min(max_lv, n_features, max_train_samples - 1)
    upper_lv = max(1, upper_lv)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_lv = 1
    best_rmse_cv = float('inf')
    
    for lv in range(1, upper_lv + 1):
        y_cv = np.zeros_like(y_train)
        try:
            for tr_idx, val_idx in kf.split(X_train):
                if model_type == 'PLSR':
                    m = PLSRegression(n_components=lv)
                elif model_type == 'OPLS':
                    m = OPLS(n_filter=1, n_pls=lv)
                elif model_type == 'KPLS':
                    m = make_pipeline(
                        Nystroem(kernel='rbf', gamma=0.001, n_components=min(100, n_features), random_state=42),
                        PLSRegression(n_components=lv)
                    )
                elif model_type == 'iPLS':
                    m = iPLS(intervals=10, n_pls=lv, n_splits=n_splits)
                
                m.fit(X_train[tr_idx], y_train[tr_idx])
                y_cv[val_idx] = m.predict(X_train[val_idx])
            
            rmse_cv = np.sqrt(mean_squared_error(y_train, y_cv))
            if rmse_cv < best_rmse_cv:
                best_rmse_cv = rmse_cv
                best_lv = lv
        except Exception:
            continue

    if model_type == 'PLSR':
        best_model = PLSRegression(n_components=best_lv)
    elif model_type == 'OPLS':
        best_model = OPLS(n_filter=1, n_pls=best_lv)
    elif model_type == 'KPLS':
        best_model = make_pipeline(
            Nystroem(kernel='rbf', gamma=0.001, n_components=min(100, n_features), random_state=42),
            PLSRegression(n_components=best_lv)
        )
    elif model_type == 'iPLS':
        best_model = iPLS(intervals=10, n_pls=best_lv, n_splits=n_splits)

    best_model.fit(X_train, y_train)
    
    # Calibration evaluation
    y_c = best_model.predict(X_train)
    r2_c, rmse_c, rpd_c = calc_metrics(y_train, y_c)

    # Validation evaluation (5-Fold CV)
    y_v = np.zeros_like(y_train)
    for tr_idx, val_idx in kf.split(X_train):
        m_eval = copy.deepcopy(best_model)
        m_eval.fit(X_train[tr_idx], y_train[tr_idx])
        y_v[val_idx] = m_eval.predict(X_train[val_idx])
    r2_v, rmse_v, rpd_v = calc_metrics(y_train, y_v)

    return best_model, best_lv, (r2_c, rmse_c, rpd_c), (r2_v, rmse_v, rpd_v)

def format_mean_std(values):
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{mean_val:.4f}|{std_val:.4f}"

# =========================================================
# 4. Table Builders Matching Attached Templates
# =========================================================
def build_file1_summary_table(rows, models, metric_mode='test'):
    # Exactly matching the structure of 2.Multi-Regression control.xlsx:
    # Row 1: ['controlle', '', '', 'PLSR', '', '', '', 'iPLSR', '', '', '', 'OPLSR', '', '', '', 'KPLSR', '', '', '']
    # Row 2: ['', '', 'Number of features', 'R2', 'RMSE', 'RPD', 'LV', 'R2', 'RMSE', 'RPD', 'LV', 'R2', 'RMSE', 'RPD', 'LV', 'R2', 'RMSE', 'RPD', 'LV']
    header_row_1 = ['controlle', '', '']
    header_row_2 = ['', '', 'Number of features']
    
    for m in models:
        header_row_1.extend([m, '', '', ''])
        header_row_2.extend(['R2', 'RMSE', 'RPD', 'LV'])

    table_data = [header_row_1, header_row_2]
    for row in rows:
        row_cells = [
            row['Group'],
            row['Target'],
            row['Number of features']
        ]
        for m in models:
            for metric in ['R2', 'RMSE', 'RPD', 'LV']:
                key = f"{m}_{metric}_{metric_mode}" if metric != 'LV' else f"{m}_LV"
                row_cells.append(row.get(key, ''))
        table_data.append(row_cells)
        
    return pd.DataFrame(table_data)

def build_file2_metric_table(models, target_cols, file_names, results_matrix):
    # Exactly matching image_df7843.png:
    # Col A: Model (PLSR, iPLSR, OPLSR, KPLSR)
    # Col B: Target (Oil (%), Moisture (%), Protein (%), Starch (%))
    # Col C...: 1.xlsx, 2.xlsx, ..., 50.xlsx
    header = ['', ''] + file_names
    table_rows = [header]
    
    for m in models:
        for t_idx, target in enumerate(target_cols):
            model_label = m if t_idx == 0 else ''
            row = [model_label, target]
            for fname in file_names:
                val = results_matrix[m][target].get(fname, np.nan)
                row.append(round(val, 5) if not np.isnan(val) else '')
            table_rows.append(row)
            
    return pd.DataFrame(table_rows)

# =========================================================
# 5. Main Execution Pipeline
# =========================================================
def main():
    train_dir = r"E:\Artical\6-EN-Explainable AI\Train-Preprocess"
    test_dir = r"E:\Artical\6-EN-Explainable AI\Test-Preprocess"
    output_dir = r"E:\Artical\6-EN-Explainable AI"

    output_file_1 = os.path.join(output_dir, "2.Multi-Regression control.xlsx")
    output_file_2 = os.path.join(output_dir, "2.Test_Multi-Regression control.xlsx")

    print(f"Scanning Train Directory: {train_dir}")
    print(f"Scanning Test Directory:  {test_dir}")

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.xlsx")), key=lambda x: natural_sort_key(os.path.basename(x)))
    if not train_files:
        print(f"Error: No Excel files found in {train_dir}")
        return

    test_file_candidates = glob.glob(os.path.join(test_dir, "*.xlsx"))
    test_file_map = {os.path.basename(f).lower(): f for f in test_file_candidates}

    # Match train files with their counterpart in test directory
    matched_file_pairs = []
    for tr_f in train_files:
        bname = os.path.basename(tr_f)
        if bname.lower() in test_file_map:
            matched_file_pairs.append((bname, tr_f, test_file_map[bname.lower()]))
        else:
            print(f"Warning: Corresponding test file not found for {bname}. Skipping.")

    if not matched_file_pairs:
        print("Error: No matched train-test file pairs found.")
        return

    print(f"Successfully matched {len(matched_file_pairs)} train-test file pairs.")

    models = ['PLSR', 'iPLSR', 'OPLSR', 'KPLSR']
    model_name_map = {'PLSR': 'PLSR', 'iPLSR': 'iPLS', 'OPLSR': 'OPLS', 'KPLSR': 'KPLS'}

    # Read the first train file to determine target column names and spectral features count
    sample_df = pd.read_excel(matched_file_pairs[0][1])
    target_cols = [str(c).strip() for c in sample_df.columns[-4:]]
    spectral_cols = [str(c).strip() for c in sample_df.columns[1:-4]]
    num_features = len(spectral_cols)

    print(f"Detected {num_features} spectral features (Control - full spectrum).")
    print(f"Targets: {target_cols}")

    # Structures to collect metric values across all 50 files
    # file2_data[metric][model][target][file_name] = value
    file2_data = {
        'R2': {m: {t: {} for t in target_cols} for m in models},
        'RMSE': {m: {t: {} for t in target_cols} for m in models},
        'RPD': {m: {t: {} for t in target_cols} for m in models}
    }

    # file1_stats[model][target] = {'cal': {'R2': [], 'RMSE': [], 'RPD': [], 'LV': []},
    #                              'val': {'R2': [], 'RMSE': [], 'RPD': [], 'LV': []},
    #                              'test': {'R2': [], 'RMSE': [], 'RPD': []}}
    file1_stats = {
        m: {
            t: {
                'cal': {'R2': [], 'RMSE': [], 'RPD': [], 'LV': []},
                'val': {'R2': [], 'RMSE': [], 'RPD': [], 'LV': []},
                'test': {'R2': [], 'RMSE': [], 'RPD': []}
            } for t in target_cols
        } for m in models
    }

    file_names_ordered = [pair[0] for pair in matched_file_pairs]

    for file_idx, (bname, tr_path, te_path) in enumerate(matched_file_pairs, 1):
        print("" + "="*80)
        print(f"[{file_idx}/{len(matched_file_pairs)}] Processing File Pair: {bname}")
        print("="*80)

        df_train = pd.read_excel(tr_path)
        df_test = pd.read_excel(te_path)

        X_tr = df_train.iloc[:, 1:-4].values.astype(float)
        X_te = df_test.iloc[:, 1:-4].values.astype(float)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for t_idx, target_name in enumerate(target_cols):
            y_train = df_train.iloc[:, -4 + t_idx].values.astype(float)
            y_test = df_test.iloc[:, -4 + t_idx].values.astype(float)

            for m_key in models:
                internal_mtype = model_name_map[m_key]
                print(f"Training: {m_key:<6} | Target: {target_name:<15} | File: {bname} | CV: 5-Fold")

                best_model, best_lv, cal_metrics, val_metrics = fit_and_cross_validate(
                    internal_mtype, X_tr_sc, y_train, max_lv=15, n_splits=5
                )

                # Test evaluation on the corresponding matched test file
                y_pred_te = best_model.predict(X_te_sc)
                r2_t, rmse_t, rpd_t = calc_metrics(y_test, y_pred_te)

                # Store for File 2
                file2_data['R2'][m_key][target_name][bname] = r2_t
                file2_data['RMSE'][m_key][target_name][bname] = rmse_t
                file2_data['RPD'][m_key][target_name][bname] = rpd_t

                # Store for File 1 statistics aggregation
                file1_stats[m_key][target_name]['cal']['R2'].append(cal_metrics[0])
                file1_stats[m_key][target_name]['cal']['RMSE'].append(cal_metrics[1])
                file1_stats[m_key][target_name]['cal']['RPD'].append(cal_metrics[2])
                file1_stats[m_key][target_name]['cal']['LV'].append(best_lv)

                file1_stats[m_key][target_name]['val']['R2'].append(val_metrics[0])
                file1_stats[m_key][target_name]['val']['RMSE'].append(val_metrics[1])
                file1_stats[m_key][target_name]['val']['RPD'].append(val_metrics[2])
                file1_stats[m_key][target_name]['val']['LV'].append(best_lv)

                file1_stats[m_key][target_name]['test']['R2'].append(r2_t)
                file1_stats[m_key][target_name]['test']['RMSE'].append(rmse_t)
                file1_stats[m_key][target_name]['test']['RPD'].append(rpd_t)

    # =========================================================
    # Build & Export File 1: 2.Multi-Regression control.xlsx
    # Sheets: Calibration_Results, Validation_Results, Test_Summary_Mean_Std
    # =========================================================
    summary_rows = []
    for t_idx, target_name in enumerate(target_cols):
        row_dict = {
            'Group': 'controlle' if t_idx == 0 else '',
            'Target': target_name,
            'Number of features': num_features
        }
        for m_key in models:
            # Cal stats
            c_r2 = file1_stats[m_key][target_name]['cal']['R2']
            c_rmse = file1_stats[m_key][target_name]['cal']['RMSE']
            c_rpd = file1_stats[m_key][target_name]['cal']['RPD']
            c_lv = file1_stats[m_key][target_name]['cal']['LV']

            row_dict[f'{m_key}_R2_cal'] = format_mean_std(c_r2)
            row_dict[f'{m_key}_RMSE_cal'] = format_mean_std(c_rmse)
            row_dict[f'{m_key}_RPD_cal'] = format_mean_std(c_rpd)
            row_dict[f'{m_key}_LV'] = int(round(np.mean(c_lv)))

            # Val stats
            v_r2 = file1_stats[m_key][target_name]['val']['R2']
            v_rmse = file1_stats[m_key][target_name]['val']['RMSE']
            v_rpd = file1_stats[m_key][target_name]['val']['RPD']

            row_dict[f'{m_key}_R2_val'] = format_mean_std(v_r2)
            row_dict[f'{m_key}_RMSE_val'] = format_mean_std(v_rmse)
            row_dict[f'{m_key}_RPD_val'] = format_mean_std(v_rpd)

            # Test stats
            t_r2 = file1_stats[m_key][target_name]['test']['R2']
            t_rmse = file1_stats[m_key][target_name]['test']['RMSE']
            t_rpd = file1_stats[m_key][target_name]['test']['RPD']

            row_dict[f'{m_key}_R2_test'] = format_mean_std(t_r2)
            row_dict[f'{m_key}_RMSE_test'] = format_mean_std(t_rmse)
            row_dict[f'{m_key}_RPD_test'] = format_mean_std(t_rpd)

        summary_rows.append(row_dict)

    df_file1_cal = build_file1_summary_table(summary_rows, models, metric_mode='cal')
    df_file1_val = build_file1_summary_table(summary_rows, models, metric_mode='val')
    df_file1_test = build_file1_summary_table(summary_rows, models, metric_mode='test')

    with pd.ExcelWriter(output_file_1, engine='openpyxl') as writer:
        df_file1_cal.to_excel(writer, sheet_name='Calibration_Results', header=False, index=False)
        df_file1_val.to_excel(writer, sheet_name='Validation_Results', header=False, index=False)
        df_file1_test.to_excel(writer, sheet_name='Test_Summary_Mean_Std', header=False, index=False)

    print(f"[File 1 Saved Successfully]: {output_file_1}")

    # =========================================================
    # Build & Export File 2: 2.Test_Multi-Regression control.xlsx
    # Sheets: R2, RMSE, RPD (Matching image_df7843.png layout)
    # =========================================================
    df_file2_r2 = build_file2_metric_table(models, target_cols, file_names_ordered, file2_data['R2'])
    df_file2_rmse = build_file2_metric_table(models, target_cols, file_names_ordered, file2_data['RMSE'])
    df_file2_rpd = build_file2_metric_table(models, target_cols, file_names_ordered, file2_data['RPD'])

    with pd.ExcelWriter(output_file_2, engine='openpyxl') as writer:
        df_file2_r2.to_excel(writer, sheet_name='R2', header=False, index=False)
        df_file2_rmse.to_excel(writer, sheet_name='RMSE', header=False, index=False)
        df_file2_rpd.to_excel(writer, sheet_name='RPD', header=False, index=False)

    print(f"[File 2 Saved Successfully]: {output_file_2}")
    print("" + "="*80)
    print("All tasks completed strictly according to specifications.")
    print("="*80 + "")

if __name__ == '__main__':
    main()
