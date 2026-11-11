import os
import glob
import copy
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
# 3. Evaluation & Optimization Helpers
# =========================================================
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
    
    y_c = best_model.predict(X_train)
    r2_c, rmse_c, rpd_c = calc_metrics(y_train, y_c)

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

def build_custom_table(rows, models, metric_mode='test'):
    header_row_1 = ['', '', '']
    header_row_2 = ['Feature Selection', 'Target', 'Number of features']
    
    for m in models:
        header_row_1.extend([m, '', '', ''])
        header_row_2.extend(['R2', 'RMSE', 'RPD', 'LV'])

    table_data = [header_row_1, header_row_2]
    for row in rows:
        row_cells = [
            row['Feature Selection'],
            row['Target'],
            row['Number of features']
        ]
        for m in models:
            for metric in ['R2', 'RMSE', 'RPD', 'LV']:
                key = f"{m}_{metric}_{metric_mode}" if metric != 'LV' else f"{m}_LV"
                row_cells.append(row.get(key, ''))
        table_data.append(row_cells)
        
    return pd.DataFrame(table_data)

# =========================================================
# 4. Main Execution Pipeline
# =========================================================
def main():
    train_dir = r"E:\Artical\6-EN-Explainable AI\Training-Regression Files"
    test_dir = r"E:\Artical\6-EN-Explainable AI\Test-Preprocess"
    
    summary_output_excel = r"E:\Artical\6-EN-Explainable AI\Comprehensive_Regression_Results.xlsx"
    detailed_test_output_excel = r"E:\Artical\6-EN-Explainable AI\Test_Files_Individual_Performances.xlsx"

    print(f"Scanning training directory: {train_dir}")
    print(f"Scanning test directory: {test_dir}")

    train_files = glob.glob(os.path.join(train_dir, "*.xlsx"))
    test_files = glob.glob(os.path.join(test_dir, "*.xlsx"))

    if not train_files:
        print(f"Error: No Excel files found in training directory: {train_dir}")
        return
    if not test_files:
        print(f"Error: No Excel files found in test directory: {test_dir}")
        return

    print(f"Found {len(test_files)} test files to evaluate.")

    # Preload all test files into memory
    test_dfs = []
    for tf in test_files:
        try:
            df_t = pd.read_excel(tf)
            test_dfs.append((os.path.basename(tf), df_t))
        except Exception as e:
            print(f"Warning: Failed to load test file {tf}: {e}")

    if not test_dfs:
        print("Error: None of the test files could be loaded.")
        return

    models = ['PLSR', 'iPLSR', 'OPLSR', 'KPLSR']
    model_name_map = {'PLSR': 'PLSR', 'iPLSR': 'iPLS', 'OPLSR': 'OPLS', 'KPLSR': 'KPLS'}

    summary_records = []
    individual_test_records = []

    for train_file in train_files:
        method_name = os.path.splitext(os.path.basename(train_file))[0]
        print("" + "="*80)
        print(f"Processing Feature Selection Method: {method_name}")
        print("="*80)
        
        try:
            xl_train = pd.ExcelFile(train_file)
        except Exception as e:
            print(f"Error reading file {train_file}: {e}")
            continue

        for sheet_name in xl_train.sheet_names:
            df_train = xl_train.parse(sheet_name)
            
            target_col = str(df_train.columns[-1]).strip()
            feature_cols = [str(c).strip() for c in df_train.columns[1:-1]]
            num_features = len(feature_cols)

            X_tr_raw = df_train.iloc[:, 1:-1].values.astype(float)
            y_train = df_train.iloc[:, -1].values.astype(float)

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_tr_raw)

            row_record = {
                'Feature Selection': method_name,
                'Target': target_col,
                'Number of features': num_features
            }

            for m_key in models:
                internal_mtype = model_name_map[m_key]
                print(f"Training Model: {m_key:<6} | Target: {target_col:<15} | Selected Features: {num_features}")
                
                best_model, best_lv, cal_metrics, val_metrics = fit_and_cross_validate(
                    internal_mtype, X_train_sc, y_train, max_lv=15, n_splits=5
                )

                row_record[f'{m_key}_LV'] = best_lv
                row_record[f'{m_key}_R2_cal'] = f"{cal_metrics[0]:.4f}"
                row_record[f'{m_key}_RMSE_cal'] = f"{cal_metrics[1]:.4f}"
                row_record[f'{m_key}_RPD_cal'] = f"{cal_metrics[2]:.4f}"

                row_record[f'{m_key}_R2_val'] = f"{val_metrics[0]:.4f}"
                row_record[f'{m_key}_RMSE_val'] = f"{val_metrics[1]:.4f}"
                row_record[f'{m_key}_RPD_val'] = f"{val_metrics[2]:.4f}"

                test_r2_list = []
                test_rmse_list = []
                test_rpd_list = []

                for tf_name, df_test_full in test_dfs:
                    target_columns_test = [str(c).strip() for c in df_test_full.columns[-4:]]
                    test_columns_clean = [str(c).strip() for c in df_test_full.columns]
                    test_spectral_cols = [str(c).strip() for c in df_test_full.columns[1:-4]]

                    matched_test_col = None
                    for tc in target_columns_test:
                        if tc.lower() == target_col.lower() or sheet_name.strip().lower() in tc.lower() or tc.lower() in sheet_name.strip().lower():
                            matched_test_col = tc
                            break

                    if matched_test_col is None:
                        keywords = ['oil', 'moisture', 'protein', 'starch']
                        for kw in keywords:
                            if kw in target_col.lower() or kw in sheet_name.strip().lower():
                                for tc in target_columns_test:
                                    if kw in tc.lower():
                                        matched_test_col = tc
                                        break
                                if matched_test_col:
                                    break

                    if matched_test_col is None:
                        continue

                    y_test_file = df_test_full[matched_test_col].values.astype(float)

                    feature_indices = [test_columns_clean.index(f) for f in feature_cols if f in test_columns_clean]
                    if len(feature_indices) == num_features:
                        X_te_raw = df_test_full.iloc[:, feature_indices].values.astype(float)
                    else:
                        sub_idx = [test_spectral_cols.index(f) for f in feature_cols if f in test_spectral_cols]
                        if len(sub_idx) == num_features:
                            X_te_raw = df_test_full.iloc[:, [1 + idx for idx in sub_idx]].values.astype(float)
                        else:
                            X_te_raw = df_test_full.iloc[:, 1:1+num_features].values.astype(float)

                    X_test_sc = scaler.transform(X_te_raw)
                    y_pred_t = best_model.predict(X_test_sc)
                    
                    r2_t, rmse_t, rpd_t = calc_metrics(y_test_file, y_pred_t)
                    test_r2_list.append(r2_t)
                    test_rmse_list.append(rmse_t)
                    test_rpd_list.append(rpd_t)

                    # Save raw metrics for statistical comparison (ANOVA / t-test / Cohen's d)
                    individual_test_records.append({
                        'Test_File': tf_name,
                        'Feature_Selection': method_name,
                        'Target': target_col,
                        'Model': m_key,
                        'Optimal_LV': best_lv,
                        'Number_of_Features': num_features,
                        'R2_Test': round(r2_t, 6),
                        'RMSE_Test': round(rmse_t, 6),
                        'RPD_Test': round(rpd_t, 6)
                    })

                row_record[f'{m_key}_R2_test'] = format_mean_std(test_r2_list) if test_r2_list else "NaN|NaN"
                row_record[f'{m_key}_RMSE_test'] = format_mean_std(test_rmse_list) if test_rmse_list else "NaN|NaN"
                row_record[f'{m_key}_RPD_test'] = format_mean_std(test_rpd_list) if test_rpd_list else "NaN|NaN"

            summary_records.append(row_record)

    # ---------------------------------------------------------
    # 1. Export Summary File (3 Sheets: Calibration, Validation, Test Summary)
    # ---------------------------------------------------------
    df_calibration = build_custom_table(summary_records, models, metric_mode='cal')
    df_validation = build_custom_table(summary_records, models, metric_mode='val')
    df_test_summary = build_custom_table(summary_records, models, metric_mode='test')

    with pd.ExcelWriter(summary_output_excel, engine='openpyxl') as writer:
        df_calibration.to_excel(writer, sheet_name='Calibration_Results', header=False, index=False)
        df_validation.to_excel(writer, sheet_name='Validation_Results', header=False, index=False)
        df_test_summary.to_excel(writer, sheet_name='Test_Summary_Mean_Std', header=False, index=False)

    print(f"Saved Summary Results: {summary_output_excel}")

    # ---------------------------------------------------------
    # 2. Export Separate Raw Test Performance File (For Stats & Effect Size)
    # ---------------------------------------------------------
    df_individual = pd.DataFrame(individual_test_records)
    
    with pd.ExcelWriter(detailed_test_output_excel, engine='openpyxl') as writer:
        # Full long-format master data
        df_individual.to_excel(writer, sheet_name='Raw_Long_Data', index=False)

        # Pivot tables for direct paired comparisons across algorithms
        for metric in ['RMSE_Test', 'R2_Test', 'RPD_Test']:
            pivoted = df_individual.pivot_table(
                index=['Test_File', 'Target'],
                columns=['Feature_Selection', 'Model'],
                values=metric
            )
            sheet_title = f'Matrix_{metric[:4]}'
            pivoted.to_excel(writer, sheet_name=sheet_title)

    print(f"Saved Individual Test Performances: {detailed_test_output_excel}")
    print("" + "="*80)
    print("All tasks completed successfully.")
    print("="*80 + "")

if __name__ == '__main__':
    main()
