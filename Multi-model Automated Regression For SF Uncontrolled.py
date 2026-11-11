import copy
import glob
import os
import warnings
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sd = np.std(y_true, ddof=1) if len(y_true) > 1 else np.std(y_true)
    rpd = sd / rmse if rmse > 1e-12 else np.nan
    return r2, rmse, rpd

def format_mean_std(values):
    if not values:
        return "NaN|NaN"
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{mean_val:.4f}|{std_val:.4f}"

def get_base_model(model_type, lv, n_features, n_splits=5):
    if model_type == 'PLSR':
        return PLSRegression(n_components=lv)
    elif model_type == 'OPLS':
        return OPLS(n_filter=1, n_pls=lv)
    elif model_type == 'KPLS':
        return make_pipeline(
            Nystroem(kernel='rbf', gamma=0.001, n_components=min(100, n_features), random_state=42),
            PLSRegression(n_components=lv)
        )
    elif model_type == 'iPLS':
        return iPLS(intervals=10, n_pls=lv, n_splits=n_splits)

def fit_and_cross_validate(model_type, X_train, y_train, max_lv=15, n_splits=5):
    n_samples, n_features = X_train.shape
    max_train_samples = n_samples - (n_samples // n_splits)
    upper_lv = max(1, min(max_lv, n_features, max_train_samples - 1))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_lv = 1
    best_rmse_cv = float('inf')

    for lv in range(1, upper_lv + 1):
        y_cv = np.zeros_like(y_train)
        try:
            for tr_idx, val_idx in kf.split(X_train):
                m = get_base_model(model_type, lv, n_features, n_splits)
                m.fit(X_train[tr_idx], y_train[tr_idx])
                y_cv[val_idx] = m.predict(X_train[val_idx])

            rmse_cv = np.sqrt(mean_squared_error(y_train, y_cv))
            if rmse_cv < best_rmse_cv:
                best_rmse_cv = rmse_cv
                best_lv = lv
        except Exception:
            continue

    # ارزیابی فولد به فولد بر روی بهترین LV جهت محاسبه انحراف معیار کالیبراسیون و اعتبارسنجی
    cal_r2_list, cal_rmse_list, cal_rpd_list = [], [], []
    val_r2_list, val_rmse_list, val_rpd_list = [], [], []

    for tr_idx, val_idx in kf.split(X_train):
        m_eval = get_base_model(model_type, best_lv, n_features, n_splits)
        m_eval.fit(X_train[tr_idx], y_train[tr_idx])

        # ارزیابی روی بخش آموزش فولد (Calibration)
        y_tr_pred = m_eval.predict(X_train[tr_idx])
        r2_tr, rmse_tr, rpd_tr = calc_metrics(y_train[tr_idx], y_tr_pred)
        cal_r2_list.append(r2_tr)
        cal_rmse_list.append(rmse_tr)
        cal_rpd_list.append(rpd_tr)

        # ارزیابی روی بخش تست فولد (Validation)
        y_val_pred = m_eval.predict(X_train[val_idx])
        r2_v, rmse_v, rpd_v = calc_metrics(y_train[val_idx], y_val_pred)
        val_r2_list.append(r2_v)
        val_rmse_list.append(rmse_v)
        val_rpd_list.append(rpd_v)

    # آموزش مدل نهایی بر کل مجموعه آموزش
    best_model = get_base_model(model_type, best_lv, n_features, n_splits)
    best_model.fit(X_train, y_train)

    cal_stats = (format_mean_std(cal_r2_list), format_mean_std(cal_rmse_list), format_mean_std(cal_rpd_list))
    val_stats = (format_mean_std(val_r2_list), format_mean_std(val_rmse_list), format_mean_std(val_rpd_list))

    return best_model, best_lv, cal_stats, val_stats

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

def format_and_merge_excel(ws, test_cols):
    thin_border = Border(
        left=Side(style='thin', color='D3D3D3'),
        right=Side(style='thin', color='D3D3D3'),
        top=Side(style='thin', color='D3D3D3'),
        bottom=Side(style='thin', color='D3D3D3')
    )
    align_center = Alignment(horizontal='center', vertical='center')
    header_font = Font(name='Calibri', size=11, bold=True)
    body_font = Font(name='Calibri', size=11)

    max_row = ws.max_row
    max_col = ws.max_column

    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = thin_border
            cell.font = header_font if r == 1 else body_font
            cell.alignment = align_center
            if r > 1 and c > 3 and isinstance(cell.value, (int, float)):
                cell.number_format = '0.00000'

    # ادغام سطرهای مربوط به متدهای انتخاب ویژگی (ستون A)
    current_fs = None
    start_fs_row = 2
    for r in range(2, max_row + 1):
        val = ws.cell(row=r, column=1).value
        if val != current_fs:
            if current_fs is not None and r - 1 > start_fs_row:
                ws.merge_cells(start_row=start_fs_row, start_column=1, end_row=r - 1, end_column=1)
            current_fs = val
            start_fs_row = r
    if max_row >= start_fs_row:
        ws.merge_cells(start_row=start_fs_row, start_column=1, end_row=max_row, end_column=1)

    # ادغام سطرهای مربوط به مدل‌ها (ستون B)
    current_m = None
    start_m_row = 2
    for r in range(2, max_row + 1):
        val = ws.cell(row=r, column=2).value
        if val != current_m:
            if current_m is not None and r - 1 > start_m_row:
                ws.merge_cells(start_row=start_m_row, start_column=2, end_row=r - 1, end_column=2)
            current_m = val
            start_m_row = r
    if max_row >= start_m_row:
        ws.merge_cells(start_row=start_m_row, start_column=2, end_row=max_row, end_column=2)

# =========================================================
# 4. Main Execution Pipeline
# =========================================================
def main():
    train_dir = r"E:\Artical\6-EN-Explainable AI\Training-Regression Files"
    test_dir = r"E:\Artical\6-EN-Explainable AI\Test-Preprocess"

    # تغییر نام فایل‌های خروجی
    summary_output_excel = os.path.join(os.path.dirname(train_dir), "1.Multi-Regression Uncontrolled.xlsx")
    detailed_test_output_excel = os.path.join(os.path.dirname(test_dir), "1.Test_Multi-Regression Uncontrolled.xlsx")

    print(f"Scanning training directory: {train_dir}")
    print(f"Scanning test directory: {test_dir}")

    train_files = glob.glob(os.path.join(train_dir, "*.xlsx"))
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.xlsx")))

    if not train_files:
        print(f"Error: No Excel files found in training directory: {train_dir}")
        return
    if not test_files:
        print(f"Error: No Excel files found in test directory: {test_dir}")
        return

    print(f"Found {len(test_files)} test files to evaluate.")

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
        print("\n" + "=" * 80)
        print(f"Processing Feature Selection Method: {method_name}")
        print("=" * 80)

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
                row_record[f'{m_key}_R2_cal'] = cal_metrics[0]
                row_record[f'{m_key}_RMSE_cal'] = cal_metrics[1]
                row_record[f'{m_key}_RPD_cal'] = cal_metrics[2]

                row_record[f'{m_key}_R2_val'] = val_metrics[0]
                row_record[f'{m_key}_RMSE_val'] = val_metrics[1]
                row_record[f'{m_key}_RPD_val'] = val_metrics[2]

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
                            X_te_raw = df_test_full.iloc[:, 1:1 + num_features].values.astype(float)

                    X_test_sc = scaler.transform(X_te_raw)
                    y_pred_t = best_model.predict(X_test_sc)

                    r2_t, rmse_t, rpd_t = calc_metrics(y_test_file, y_pred_t)
                    test_r2_list.append(r2_t)
                    test_rmse_list.append(rmse_t)
                    test_rpd_list.append(rpd_t)

                    individual_test_records.append({
                        'Feature_Selection': method_name,
                        'Model': m_key,
                        'Target': target_col,
                        'Test_File': tf_name,
                        'R2': r2_t,
                        'RMSE': rmse_t,
                        'RPD': rpd_t
                    })

                row_record[f'{m_key}_R2_test'] = format_mean_std(test_r2_list)
                row_record[f'{m_key}_RMSE_test'] = format_mean_std(test_rmse_list)
                row_record[f'{m_key}_RPD_test'] = format_mean_std(test_rpd_list)

            summary_records.append(row_record)

    # ---------------------------------------------------------
    # 1. ذخیره فایل اول (1.Multi-Regression Uncontrolled.xlsx)
    # ---------------------------------------------------------
    df_calibration = build_custom_table(summary_records, models, metric_mode='cal')
    df_validation = build_custom_table(summary_records, models, metric_mode='val')
    df_test_summary = build_custom_table(summary_records, models, metric_mode='test')

    with pd.ExcelWriter(summary_output_excel, engine='openpyxl') as writer:
        df_calibration.to_excel(writer, sheet_name='Calibration_Results', header=False, index=False)
        df_validation.to_excel(writer, sheet_name='Validation_Results', header=False, index=False)
        df_test_summary.to_excel(writer, sheet_name='Test_Summary_Mean_Std', header=False, index=False)

    print(f"\nSaved Summary Results: {summary_output_excel}")

    # ---------------------------------------------------------
    # 2. ذخیره فایل دوم مطابق ساختار تصویر (1.Test_Multi-Regression Uncontrolled.xlsx)
    # ---------------------------------------------------------
    df_ind = pd.DataFrame(individual_test_records)
    test_files_unique = sorted(df_ind['Test_File'].unique().tolist())

    wb = openpyxl.Workbook()
    # حذف شیت پیش‌فرض اکسل
    wb.remove(wb.active)

    metrics_list = ['R2', 'RMSE', 'RPD']

    for metric in metrics_list:
        ws = wb.create_sheet(title=metric)
        # سرستون‌ها
        headers = ['', '', ''] + test_files_unique
        ws.append(headers)

        # پیوت کردن جهت ساخت جدول بر اساس Feature Selection -> Model -> Target
        pivoted = df_ind.pivot_table(
            index=['Feature_Selection', 'Model', 'Target'],
            columns='Test_File',
            values=metric,
            sort=False
        ).reset_index()

        # بازسازی ترتیب دقیق مدل‌ها مطابق تصویر
        model_order = {m: i for i, m in enumerate(models)}
        pivoted['model_rank'] = pivoted['Model'].map(model_order)
        pivoted = pivoted.sort_values(by=['Feature_Selection', 'model_rank']).drop(columns=['model_rank'])

        for _, row in pivoted.iterrows():
            row_vals = [
                row['Feature_Selection'],
                row['Model'],
                row['Target']
            ] + [row.get(tf, '') for tf in test_files_unique]
            ws.append(row_vals)

        # استایل‌بندی، تنظیم عرض و ادغام سلول‌ها مشابه تصویر
        format_and_merge_excel(ws, test_files_unique)

    wb.save(detailed_test_output_excel)
    print(f"Saved Test Individual Performances: {detailed_test_output_excel}")
    print("\n" + "=" * 80)
    print("All tasks completed successfully.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
