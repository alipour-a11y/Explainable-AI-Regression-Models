
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import warnings
import time


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(r"E:\Artical\6-EN-Explainable AI")

TRAIN_DIR = BASE_DIR / "Train"
TEST_DIR = BASE_DIR / "Test"

TRAIN_OUTPUT_DIR = BASE_DIR / "Train-Preprocess"
TEST_OUTPUT_DIR = BASE_DIR / "Test-Preprocess"

SG_WINDOW = 11
SG_POLYORDER = 2

BL_POLYORDER = 2


# ============================================================
# 2. SAVITZKY-GOLAY
# Exactly as in the supplied code
# ============================================================

def get_sg_parameters(n_features, derivative=0):

    window = min(SG_WINDOW, n_features)

    if window % 2 == 0:
        window -= 1

    min_window = max(SG_POLYORDER + 1, derivative + 1)

    if min_window % 2 == 0:
        min_window += 1

    if window < min_window:
        raise ValueError(
            f"Not enough spectral features: {n_features}"
        )

    return window


def apply_sg(X, derivative=0):

    window = get_sg_parameters(
        X.shape[1], derivative
    )

    return savgol_filter(
        X,
        window_length=window,
        polyorder=SG_POLYORDER,
        deriv=derivative,
        axis=1,
        mode="interp"
    )


# ============================================================
# 3. POLYNOMIAL BASELINE CORRECTION
# Exactly as in the supplied code
# ============================================================

def apply_baseline(X):

    X = np.asarray(X, dtype=np.float64)

    n_samples, n_features = X.shape

    # Normalize wavelength/index axis for numerical stability
    wavelength = np.linspace(-1, 1, n_features)

    # Polynomial design matrix
    V = np.vander(
        wavelength,
        N=BL_POLYORDER + 1,
        increasing=True
    )

    # Estimate polynomial coefficients for all spectra
    coefficients = np.linalg.lstsq(
        V,
        X.T,
        rcond=None
    )[0]

    # Calculate baseline
    baseline = (V @ coefficients).T

    # Subtract baseline
    corrected = X - baseline

    return corrected


# ============================================================
# 4. SG + BL PREPROCESSING
# ============================================================

def preprocess_spectra(X):

    X = np.asarray(X, dtype=np.float64)

    # Step 1: Savitzky-Golay smoothing
    X = apply_sg(X, derivative=0)

    # Step 2: Polynomial baseline correction
    X = apply_baseline(X)

    return X


# ============================================================
# 5. PROCESS ONE DATAFRAME
# ============================================================

def process_dataframe(df, file_name, sheet_name):

    if df.shape[1] < 6:
        raise ValueError(
            f"{file_name} / {sheet_name}: "
            "At least 6 columns are required."
        )

    # First column: sample identifiers
    sample_names = df.iloc[:, 0].copy()

    # Middle columns: spectral features
    spectral_df = df.iloc[:, 1:-4].copy()

    # Last four columns: response variables
    response_df = df.iloc[:, -4:].copy()

    # Convert spectral columns to numeric
    spectral_df = spectral_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if spectral_df.isna().any().any():

        raise ValueError(
            f"{file_name} / {sheet_name}: "
            "Spectral features contain missing "
            "or non-numeric values."
        )

    X = spectral_df.to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():

        raise ValueError(
            f"{file_name} / {sheet_name}: "
            "Spectral features contain non-finite values."
        )

    # Apply SG + BL
    X_processed = preprocess_spectra(X)

    # Reconstruct processed spectral dataframe
    processed_spectral_df = pd.DataFrame(
        X_processed,
        columns=spectral_df.columns,
        index=df.index
    )

    # Combine original sample names, processed spectra,
    # and original response variables
    output_df = pd.concat(
        [
            sample_names,
            processed_spectral_df,
            response_df
        ],
        axis=1
    )

    return output_df


# ============================================================
# 6. PROCESS ALL EXCEL FILES IN A FOLDER
# ============================================================

def process_folder(input_dir, output_dir):

    if not input_dir.exists():

        print(
            f"\n[ERROR] Folder not found: {input_dir}",
            flush=True
        )

        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Numeric Excel filenames, matching the supplied workflow
    files = sorted(
        [
            p for p in input_dir.glob("*.xlsx")
            if p.stem.isdigit()
            and not p.name.startswith("~$")
        ],
        key=lambda p: int(p.stem)
    )

    if not files:

        print(
            f"\nNo numeric Excel files found in {input_dir}",
            flush=True
        )

        return

    print("\n" + "=" * 75)
    print(f"FOLDER: {input_dir.name}")
    print(f"TOTAL FILES: {len(files)}")
    print("=" * 75)

    success_count = 0
    error_count = 0

    for file_number, file_path in enumerate(files, 1):

        output_path = output_dir / file_path.name

        print(
            f"\n[{file_number}/{len(files)}] "
            f"Processing: {file_path.name}",
            flush=True
        )

        try:

            # Read all worksheets
            excel_file = pd.ExcelFile(file_path)

            with pd.ExcelWriter(
                output_path,
                engine="openpyxl"
            ) as writer:

                for sheet_name in excel_file.sheet_names:

                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name
                    )

                    processed_df = process_dataframe(
                        df,
                        file_path.name,
                        sheet_name
                    )

                    processed_df.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False
                    )

            success_count += 1

            print(
                f"   SUCCESS: {output_path.name}",
                flush=True
            )

        except Exception as e:

            error_count += 1

            print(
                f"   ERROR: {file_path.name}",
                flush=True
            )

            print(
                f"   Reason: {e}",
                flush=True
            )

            # Remove incomplete output
            if output_path.exists():
                output_path.unlink()

    print("\n" + "-" * 75)
    print(f"Folder completed: {input_dir.name}")
    print(f"Successful files: {success_count}")
    print(f"Failed files: {error_count}")
    print("-" * 75)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    start_time = time.time()

    print("\n" + "=" * 75)
    print("SG + POLYNOMIAL BL PREPROCESSING")
    print("=" * 75)

    print(f"SG Window       : {SG_WINDOW}")
    print(f"SG Polynomial   : {SG_POLYORDER}")
    print(f"BL Polynomial   : {BL_POLYORDER}")
    print("Processing order: SG -> BL")

    # First: Train
    process_folder(
        TRAIN_DIR,
        TRAIN_OUTPUT_DIR
    )

    # Second: Test
    process_folder(
        TEST_DIR,
        TEST_OUTPUT_DIR
    )

    elapsed_time = (time.time() - start_time) / 60

    print("\n" + "=" * 75)
    print("ALL PREPROCESSING TASKS COMPLETED")
    print(f"Execution time: {elapsed_time:.2f} minutes")
    print("=" * 75)
