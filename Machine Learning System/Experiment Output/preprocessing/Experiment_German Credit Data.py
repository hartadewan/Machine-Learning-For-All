# %% [markdown]
# # **1. Dataset Introduction**

# %% [markdown]
# 1. **Dataset Source**:
# The experiments in this repository use the Statlog (German Credit Data) from the UCI Machine Learning Repository. This dataset classifies individuals described by a number of attributes as either good or bad credit risks. The original data was provided by Prof. Hofmann and contains categorical/symbolic attributes in the file “german.data”. The dataset was donated on November 16, 1994.

# %% [markdown]
# ## **2. Import Library**

# %%
import warnings
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----- Konstanta -----
RANDOM_STATE = 42
TEST_SIZE = 0.20


SEABORN_DEEP_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def configure_environment() -> None:
    warnings.filterwarnings("ignore")

    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 120)


def report_versions() -> None:
    print("Seluruh pustaka berhasil diimpor.")
    print(f"Versi pandas       : {pd.__version__}")
    print(f"Versi numpy        : {np.__version__}")
    print(f"Versi scikit-learn : {sklearn.__version__}")


configure_environment()
report_versions()

# %% [markdown]
# # **3. Dataset Load**

# %%
COLUMN_NAMES = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property_type",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target",
]

def resolve_raw_path(filename="credit_scoring_raw.csv"):
    candidates = [
        Path(filename),
        Path("..") / filename,
        Path("..") / "namadataset_raw" / filename,
        Path("namadataset_raw") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"'{filename}' tidak ditemukan. Lokasi yang dicek: "
        + ", ".join(str(c) for c in candidates)
    )

RAW_PATH = resolve_raw_path()
df = pd.read_csv(RAW_PATH, header=None, names=COLUMN_NAMES)

print(f"Dataset dimuat dari: {RAW_PATH}")
print(f"Dimensi            : {df.shape[0]} baris x {df.shape[1]} kolom")
df.head()


# %%
# -------- Struktur & tipe data --------
df.info()


# %% [markdown]
# # **4. Exploratory Data Analysis (EDA)**
# 

# %% [markdown]
# At this stage, you will do **Exploratory Data Analysis (EDA)** to understand the characteristics of the dataset.
# 
# The goal of EDA is to get an initial deep insight into the data and decide the next steps in analysis or modeling.

# %%
# -------- Data quality: empty & duplicate values --------
missing = df.isna().sum()
print("Total nilai kosong :", int(missing.sum()))
print("Total baris duplikat:", int(df.duplicated().sum()))

# ----- Descriptive statistics -----
df.describe().T


# %%
# -------- Target distribution -------
target_label = df["target"].map({1: "Good (1)", 2: "Bad (2)"})
counts = target_label.value_counts()
pct = (counts / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=counts.index, y=counts.values, ax=ax)
for i, (n, p) in enumerate(zip(counts.values, pct.values)):
    ax.text(i, n + 8, f"{n} ({p}%)", ha="center", fontweight="bold")
ax.set_title("Target Distribution — Credit Risk")
ax.set_xlabel("Class"); ax.set_ylabel("Jumlah nasabah")
plt.tight_layout(); plt.show()

# %%
# ------ Distribution -------
NUMERIC = ["duration_months", "credit_amount", "installment_rate",
           "residence_since", "age_years", "existing_credits", "num_dependents"]

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.ravel(), NUMERIC):
    sns.histplot(df[col], kde=True, ax=ax, bins=25)
    ax.set_title(col)
for ax in axes.ravel()[len(NUMERIC):]:
    ax.set_visible(False)
plt.suptitle("Numerical Distribution", y=1.01, fontsize=14)
plt.tight_layout(); plt.show()


# %%
# ----- Outlier Detection -----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["credit_amount", "duration_months", "age_years"]):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.suptitle("Boxplot — Outlier Testing", y=1.03, fontsize=13)
plt.tight_layout(); plt.show()


# %%
# ─--- Feature relationship  -----
key_cats = ["checking_status", "credit_history", "savings_status"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, key_cats):
    sns.countplot(data=df.assign(Risk=target_label),
                  x=col, hue="Risk", ax=ax)
    ax.set_title(col); ax.tick_params(axis="x", rotation=45)
plt.suptitle("Fitur Kategorikal Kunci vs Credit Risk", y=1.03, fontsize=14)
plt.tight_layout(); plt.show()


# %%
# ---- Correlation matrix ----
corr_df = df[NUMERIC].copy()
corr_df["is_bad"] = (df["target"] == 2).astype(int)
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=.5, ax=ax)
ax.set_title("Numerical Correlation")
plt.tight_layout(); plt.show()


# %% [markdown]
# # **5. Data Preprocessing**

# %% [markdown]
# At this stage, data preprocessing is an important step to ensure the quality of the data before using it in a machine learning model.
# 
# If you are using text data, raw data often contains empty values, duplicates, or inconsistent ranges, which can affect the model's performance. Therefore, this process aims to clean and prepare the data so that analysis can run smoothly.

# %%
# ---- Mapping target to credit risk ----
data = df.copy()
data["credit_risk"] = data["target"].map({1: 0, 2: 1})
data = data.drop(columns=["target"])

print("Credit Risk Distribution (0=good, 1=bad):")
print(data["credit_risk"].value_counts().sort_index().to_dict())


# %%
# ---- Handling missing values & duplicates ----
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)
print(f"Baris duplikat dihapus : {before - len(data)}")

assert data.isna().sum().sum() == 0, "Masih terdapat nilai kosong!"
print("Nilai kosong tersisa   :", int(data.isna().sum().sum()))


# %%
# ---- Feature/target separation & column type assignment ----
TARGET = "credit_risk"
categorical_cols = data.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in NUMERIC if c in data.columns]

X = data.drop(columns=[TARGET])
y = data[TARGET]

print(f"Fitur numerik    ({len(numeric_cols)}): {numeric_cols}")
print(f"Fitur kategorikal ({len(categorical_cols)}): {categorical_cols}")
print(f"X: {X.shape} | y: {y.shape}")


# %%
# ---- Stratified data splitting ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Set latih : {X_train.shape[0]} baris")
print(f"Set uji   : {X_test.shape[0]} baris")
print("Proporsi bad — latih: {:.3f} | uji: {:.3f}".format(
    y_train.mean(), y_test.mean()))


# %%
# ---- Standardization + One-Hot Encoding ----
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# fit ONLY on the training set, then transform both sets
X_train_arr = preprocessor.fit_transform(X_train)
X_test_arr = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_train_prep = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
X_test_prep = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

print(f"Jumlah fitur setelah encoding: {X_train_prep.shape[1]} "
      f"(dari {X_train.shape[1]} fitur asli)")
X_train_prep.head()


# %%
# ---- Assemble train data ready & save ----
train_ready = X_train_prep.copy()
train_ready[TARGET] = y_train.values
test_ready = X_test_prep.copy()
test_ready[TARGET] = y_test.values

OUT_DIR = Path("credit_scoring_preprocessing")
OUT_DIR.mkdir(exist_ok=True)
train_ready.to_csv(OUT_DIR / "train.csv", index=False)
test_ready.to_csv(OUT_DIR / "test.csv", index=False)

# Final validation of training-ready data
assert train_ready.isna().sum().sum() == 0
assert test_ready.isna().sum().sum() == 0
print("Artefak preprocessing tersimpan di:", OUT_DIR.resolve())
print(f"  • train.csv : {train_ready.shape[0]} baris x {train_ready.shape[1]} kolom")
print(f"  • test.csv  : {test_ready.shape[0]} baris x {test_ready.shape[1]} kolom")
print("Validasi: tidak ada nilai kosong pada data siap latih.")
train_ready.head()



