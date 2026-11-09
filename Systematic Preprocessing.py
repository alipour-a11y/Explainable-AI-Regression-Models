
import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# تنظیمات اصلی
# ==================================================

INPUT_FILE = Path(r"E:\Artical\6-EN-Explainable AI\Corn.xlsx")

TRAIN_PERCENT = 80       # درصد نمونه‌های آموزش
N_REPEATS = 50           # تعداد تکرار تقسیم داده‌ها
RANDOM_SEED = 42         # بذر تصادفی برای تکرارپذیری

# ==================================================
# ایجاد مسیرهای خروجی
# ==================================================

OUTPUT_DIR = INPUT_FILE.parent
TRAIN_DIR = OUTPUT_DIR / "Train"
TEST_DIR = OUTPUT_DIR / "Test"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# خواندن فایل اکسل
# ==================================================

df = pd.read_excel(INPUT_FILE)

n_samples = len(df)

if n_samples < 2:
    raise ValueError("تعداد نمونه‌ها برای تقسیم داده کافی نیست.")

if not 0 < TRAIN_PERCENT < 100:
    raise ValueError("درصد آموزش باید بین صفر و صد باشد.")

# تعداد ثابت نمونه‌های آموزش و آزمون
n_train = int(round(n_samples * TRAIN_PERCENT / 100))
n_test = n_samples - n_train

if n_train == 0 or n_test == 0:
    raise ValueError("تعداد نمونه‌های آموزش یا آزمون صفر شده است.")

print("=" * 55)
print("DATA SPLITTING")
print("=" * 55)
print(f"Total samples : {n_samples}")
print(f"Train samples : {n_train}")
print(f"Test samples  : {n_test}")
print(f"Repeats       : {N_REPEATS}")
print("=" * 55)

# ==================================================
# تقسیم تصادفی با جلوگیری از تکرار مجموعه‌ها
# ==================================================

rng = np.random.default_rng(RANDOM_SEED)

used_train_sets = set()

for repeat in range(1, N_REPEATS + 1):

    # تولید یک تقسیم جدید و یکتا
    while True:

        shuffled_indices = rng.permutation(n_samples)

        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]

        # شناسایی مجموعه نمونه‌های آموزش
        train_key = tuple(sorted(train_indices.tolist()))

        if train_key not in used_train_sets:
            used_train_sets.add(train_key)
            break

    # استخراج نمونه‌ها
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # مرتب‌سازی بر اساس ترتیب اولیه نمونه‌ها
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    # نام‌گذاری فایل‌ها
    train_file = TRAIN_DIR / f"{repeat:02d}.xlsx"
    test_file = TEST_DIR / f"{repeat:02d}.xlsx"

    # ذخیره فایل‌ها
    train_df.to_excel(train_file, index=False)
    test_df.to_excel(test_file, index=False)

    print(
        f"Repeat {repeat:02d}/{N_REPEATS} | "
        f"Train: {len(train_df)} | "
        f"Test: {len(test_df)} | "
        f"Saved successfully"
    )

# ==================================================
# گزارش نهایی
# ==================================================

print("\n" + "=" * 55)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 55)

print(f"Train folder: {TRAIN_DIR}")
print(f"Test folder : {TEST_DIR}")

print(f"\nTotal Train files: {len(list(TRAIN_DIR.glob('*.xlsx')))}")
print(f"Total Test files : {len(list(TEST_DIR.glob('*.xlsx')))}")

print("All splits have equal sample counts and unique")
print("training sample sets.")
