import os
import re
import time
import warnings
from collections import Counter
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plot from displaying on screen
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Paths and Parameters Configuration ---
input_dir = r"E:\Artical\6-EN-Explainable AI\Train-Preprocess"
output_dir = r"E:\Artical\6-EN-Explainable AI\Train-Preprocess-SF"
os.makedirs(output_dir, exist_ok=True)

# --- Feature Selection Algorithms ---

def select_features_mrmr(X, y, max_k=50):
    n_feat = X.shape[1]
    max_k = min(max_k, n_feat)
    relevances = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_feat)])
    relevances[np.isnan(relevances)] = 0
    selected = [np.argmax(relevances)]
    for _ in range(max_k - 1):
        unselected = [i for i in range(n_feat) if i not in selected]
        scores = []
        for j in unselected:
            rel = relevances[j]
            red = np.mean([np.abs(np.corrcoef(X[:, j], X[:, s])[0, 1]) for s in selected])
            if np.isnan(red):
                red = 0
            score = rel - red
            scores.append((j, score))
        if not scores:
            break
        best_j = max(scores, key=lambda x: x[1])[0]
        selected.append(best_j)
    return np.array(sorted(selected))

def select_features_spa(X, y, max_k=50):
    n_feat = X.shape[1]
    max_k = min(max_k, n_feat)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    selected_columns = []
    corrs = [np.abs(np.corrcoef(X_scaled[:, i], y)[0, 1]) for i in range(n_feat)]
    first_col = np.argmax(corrs)
    selected_columns.append(first_col)
    current_set = set(selected_columns)
    for _ in range(max_k - 1):
        unselected = [i for i in range(n_feat) if i not in current_set]
        projections = []
        for j in unselected:
            X_sel = X_scaled[:, list(current_set)]
            xj = X_scaled[:, j]
            try:
                beta, _, _, _ = np.linalg.lstsq(X_sel, xj, rcond=None)
                res = xj - X_sel @ beta
                projections.append((j, np.linalg.norm(res)))
            except:
                projections.append((j, 0))
        if not projections:
            break
        best_j = max(projections, key=lambda x: x[1])[0]
        current_set.add(best_j)
    return np.array(sorted(list(current_set)))

def select_features_cars(X, y, n_iter=30):
    n_samp, n_feat = X.shape
    np.random.seed(42)
    current_features = np.arange(n_feat)
    for it in range(min(n_iter, n_feat)):
        if len(current_features) < 5:
            break
        X_sub = X[:, current_features]
        pls = PLSRegression(n_components=min(3, len(current_features), n_samp - 1))
        pls.fit(X_sub, y)
        coefs = np.abs(pls.coef_.flatten())
        r = (1 - (it + 1) / n_iter) * 0.7 + 0.3
        k_keep = max(5, int(len(current_features) * r))
        top_idx = np.argsort(coefs)[-k_keep:]
        current_features = current_features[top_idx]
    return np.sort(current_features)

def select_features_boruta(X, y):
    np.random.seed(42)
    n_samp, n_feat = X.shape
    shadow_X = np.apply_along_axis(np.random.permutation, 0, X)
    X_ext = np.hstack([X, shadow_X])
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_ext, y)
    importances = rf.feature_importances_
    orig_imp = importances[:n_feat]
    shadow_imp = importances[n_feat:]
    max_shadow_imp = np.max(shadow_imp)
    selected_idx = np.where(orig_imp > max_shadow_imp)[0]
    if len(selected_idx) < 5:
        selected_idx = np.argsort(orig_imp)[-15:]
    return selected_idx

# --- Active Methods Configuration (Uncomment others when ready) ---
methods = {
    "mRMR": select_features_mrmr,
    "SPA": select_features_spa,
    "CARS": select_features_cars,
    "Boruta": select_features_boruta
}

# --- Read Input Files and Preload Data ---
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.xlsx', '.xls'))])
if not all_files:
    raise ValueError(f"No Excel files found in the directory: {input_dir}")

overall_start_time = time.time()
execution_times = {}

print("[*] Loading input files and computing overall mean spectrum...")
loaded_datasets = {}
spectra_means = []

for file_name in all_files:
    file_path = os.path.join(input_dir, file_name)
    try:
        df = pd.read_excel(file_path)
        loaded_datasets[file_name] = df
        spectra_means.append(df.iloc[:, 1:-4].values.mean(axis=0))
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")

if not loaded_datasets:
    raise ValueError("No valid Excel datasets could be loaded.")

total_files_count = len(loaded_datasets)

# Reference dataset for structural metadata and consensus export
ref_file_name = list(loaded_datasets.keys())[0]
ref_df = loaded_datasets[ref_file_name]
ref_sample_names = ref_df.iloc[:, 0]
ref_x_df = ref_df.iloc[:, 1:-4]
ref_y_df = ref_df.iloc[:, -4:]

feature_names = ref_x_df.columns.tolist()
response_names = ref_y_df.columns.tolist()
overall_mean_spectrum = np.mean(spectra_means, axis=0)

# Helper function to convert feature column names to numeric wavelengths for sorting/plotting
def parse_wavelength(val):
    try:
        return float(val)
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(nums[0]) if nums else np.nan

wavelengths = np.array([parse_wavelength(f) for f in feature_names])
if np.isnan(wavelengths).any():
    wavelengths = np.arange(len(feature_names), dtype=float)
feat_to_wl = dict(zip(feature_names, wavelengths))

# Dictionary to store final consensus features: {resp_name: {method_abbr: [sorted_features]}}
final_consensus_dict = {resp: {} for resp in response_names}

# --- Execute Feature Selection ---
for method_abbr, func in methods.items():
    method_start_time = time.time()
    print(f"[*] Processing algorithm: [{method_abbr}] across all files...")
    
    method_file_selections = {resp: {} for resp in response_names}
    
    for file_name, df in loaded_datasets.items():
        base_name = os.path.splitext(file_name)[0]
        sample_names = df.iloc[:, 0]
        y_df = df.iloc[:, -4:]
        x_df = df.iloc[:, 1:-4]
        
        X = x_df.values
        Y = y_df.values
        n_samples, n_features = X.shape
        
        out_file = os.path.join(output_dir, f"{base_name}-{method_abbr}.xlsx")
        
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            for r_idx, resp_name in enumerate(response_names):
                y_curr = Y[:, r_idx]
                
                try:
                    selected_indices = func(X, y_curr)
                except Exception:
                    selected_indices = np.arange(min(20, n_features))
                    
                selected_feat_names = [feature_names[i] for i in selected_indices]
                method_file_selections[resp_name][base_name] = selected_feat_names
                
                X_sel = X[:, selected_indices]
                sheet_data = pd.DataFrame(X_sel, columns=selected_feat_names)
                sample_col_name = sample_names.name if sample_names.name else "Sample_Name"
                sheet_data.insert(0, sample_col_name, sample_names.values)
                sheet_data[resp_name] = y_curr
                
                sheet_data.to_excel(writer, sheet_name=str(resp_name), index=False)

    # --- Save Consensus File AND Frequency Statistics File ---
    consensus_out_file = os.path.join(output_dir, f"Consensus_Features_{method_abbr}.xlsx")
    stats_out_file = os.path.join(output_dir, f"Feature_Frequency_Stats_{method_abbr}.xlsx")
    
    with pd.ExcelWriter(consensus_out_file, engine='openpyxl') as consensus_writer, \
         pd.ExcelWriter(stats_out_file, engine='openpyxl') as stats_writer:
        
        for r_idx, resp_name in enumerate(response_names):
            files_dict = method_file_selections.get(resp_name, {})
            feature_counts = Counter()
            for feats in files_dict.values():
                feature_counts.update(feats)
            
            # 1. Build and save Frequency Statistics sheet (Feature, Repetition_Count, Frequency_Ratio)
            if feature_counts:
                stats_df = pd.DataFrame({
                    "Feature": list(feature_counts.keys()),
                    "Repetition_Count": list(feature_counts.values()),
                    "Frequency_Ratio": [count / total_files_count for count in feature_counts.values()]
                }).sort_values(by=["Repetition_Count", "Feature"], ascending=[False, True])
            else:
                stats_df = pd.DataFrame(columns=["Feature", "Repetition_Count", "Frequency_Ratio"])
                
            stats_df.to_excel(stats_writer, sheet_name=str(resp_name), index=False)
            
            # 2. Extract Consensus Features using Median Threshold
            counts_arr = np.array(list(feature_counts.values())) if feature_counts else np.array([0])
            threshold = np.median(counts_arr) if len(counts_arr) > 0 else 0
            
            consensus_features = [
                feat for feat, count in feature_counts.items() 
                if count >= threshold
            ]
            
            if not consensus_features and feature_counts:
                top_feat = feature_counts.most_common(1)[0][0]
                consensus_features = [top_feat]
                
            # Sort consensus features by wavelength order
            consensus_features = sorted(consensus_features, key=lambda f: feat_to_wl.get(f, 0))
            final_consensus_dict[resp_name][method_abbr] = consensus_features
            
            # Build consensus sheet: Col 1: Sample Name, Middle: Selected Features, Last: Response Variable
            if consensus_features:
                consensus_X = ref_x_df[consensus_features].values
                consensus_sheet_data = pd.DataFrame(consensus_X, columns=consensus_features)
                sample_col_name = ref_sample_names.name if ref_sample_names.name else "Sample_Name"
                consensus_sheet_data.insert(0, sample_col_name, ref_sample_names.values)
                consensus_sheet_data[resp_name] = ref_y_df.iloc[:, r_idx].values
            else:
                consensus_sheet_data = pd.DataFrame({
                    (ref_sample_names.name if ref_sample_names.name else "Sample_Name"): ref_sample_names.values,
                    resp_name: ref_y_df.iloc[:, r_idx].values
                })
            
            consensus_sheet_data.to_excel(consensus_writer, sheet_name=str(resp_name), index=False)

    method_elapsed = time.time() - method_start_time
    execution_times[method_abbr] = method_elapsed
    print(f"    -> [{method_abbr}] completed. Output, consensus, and stats files saved ({method_elapsed:.2f} seconds).")

# --- Save Summary Table (Table 2 Format) ---
table_rows = []
active_method_names = list(methods.keys())

for resp_name in response_names:
    for m_idx, method_abbr in enumerate(active_method_names):
        feats = final_consensus_dict[resp_name].get(method_abbr, [])
        feats_str = ", ".join(str(f) for f in feats)
        table_rows.append({
            "Properties": resp_name if m_idx == 0 else "",
            "Algorithm": method_abbr,
            "Selected features": feats_str
        })

table_df = pd.DataFrame(table_rows)
table_out_file = os.path.join(output_dir, "Table_2_Selected_Features.xlsx")
table_df.to_excel(table_out_file, index=False)
print(f"[*] Summary table saved to: {table_out_file}")

# --- Generate and Save Plot (Left Axis: Spectrum Values, Right Axis: 4 Responses & Algorithms) ---
fig, ax_left = plt.subplots(figsize=(12, 7), dpi=300)
ax_right = ax_left.twinx()

# Plot overall mean spectrum on the left axis
sort_idx = np.argsort(wavelengths)
wl_sorted = wavelengths[sort_idx]
spec_sorted = overall_mean_spectrum[sort_idx]

ax_left.plot(wl_sorted, spec_sorted, color='#5B9BD5', linewidth=1.8, alpha=0.85, zorder=1)
ax_left.set_xlabel("Wavelength (nm)", fontsize=12, fontweight='bold', labelpad=10)
ax_left.set_ylabel("Mean Spectral Value", fontsize=12, fontweight='bold', labelpad=10)
ax_left.grid(True, linestyle=':', alpha=0.4)

# Configure right axis for the 4 variables and active algorithms
n_resp = len(response_names)
n_alg = len(active_method_names)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors for the 4 variables

y_ticks = []
y_ticklabels = []
group_Step = n_alg + 1.2

# Plot from top to bottom
for r_idx, resp_name in enumerate(response_names):
    color = colors[r_idx % len(colors)]
    base_y = (n_resp - 1 - r_idx) * group_Step
    
    # Draw horizontal separator line between variable sections
    if r_idx > 0:
        sep_y = base_y + n_alg - 0.4
        ax_right.axhline(sep_y, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)
        
    for m_idx, method_abbr in enumerate(active_method_names):
        y_pos = base_y + (n_alg - 1 - m_idx)
        y_ticks.append(y_pos)
        y_ticklabels.append(method_abbr)
        
        sel_feats = final_consensus_dict[resp_name].get(method_abbr, [])
        sel_wls = [feat_to_wl[f] for f in sel_feats if f in feat_to_wl]
        
        # Scatter points without black borders (edgecolors='none')
        ax_right.scatter(
            sel_wls, [y_pos] * len(sel_wls),
            color=color, edgecolors='none', s=45, zorder=3
        )
        
    # Add variable group label on the right side
    group_center_y = base_y + (n_alg - 1) / 2.0
    ax_right.text(
        1.09, group_center_y, str(resp_name),
        transform=ax_right.get_yaxis_transform(),
        fontsize=12, fontweight='bold', color=color,
        va='center', ha='left', rotation=90
    )

ax_right.set_yticks(y_ticks)
ax_right.set_yticklabels(y_ticklabels, fontsize=10, fontweight='bold')
ax_right.set_ylim(-0.8, n_resp * group_Step - 0.4)

# Apply colored labels to the right y-axis algorithm ticks matching their variable group
for tick_idx, tick_label in enumerate(ax_right.get_yticklabels()):
    group_idx = tick_idx // n_alg
    tick_label.set_color(colors[group_idx % len(colors)])

plt.tight_layout()
plt.subplots_adjust(right=0.86)  # Make room on the right for variable labels

plot_out_file = os.path.join(output_dir, "Selected_Features_Plot.png")
plt.savefig(plot_out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[*] Plot saved to: {plot_out_file}")

total_elapsed = time.time() - overall_start_time

print("\n" + "="*40)
print("        EXECUTION TIME SUMMARY")
print("="*40)
for m, t in execution_times.items():
    print(f" Algorithm {m:7s}: {t:.2f} seconds")
print("-" * 40)
print(f" Total Execution Time: {total_elapsed:.2f} seconds")
print("="*40)
print(f"All outputs saved in directory: {output_dir}")