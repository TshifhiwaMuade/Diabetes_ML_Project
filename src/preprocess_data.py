from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_model_dataset.csv"
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "data" / "raw" / "cleaned_model_dataset.csv"

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

X = df.drop("diabetes_stage", axis=1)
y = df["diabetes_stage"]

# ── Feature Groups ─────────────────────────────────────────────────────────────
categorical_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status"
]

numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ── Preprocessing Pipeline ─────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# ── Train / Validation / Test Split ───────────────────────────────────────────
# 70% train | 15% validation | 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ── SMOTE — Address Class Imbalance on Training Set Only ──────────────────────
# NOTE: SMOTE is applied ONLY to training data to prevent data leakage
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ── Save Processed Splits to data/processed/ ──────────────────────────────────
output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(X_train).to_csv(output_dir / "X_train.csv", index=False)
pd.DataFrame(X_val).to_csv(output_dir / "X_val.csv", index=False)
pd.DataFrame(X_test).to_csv(output_dir / "X_test.csv", index=False)

pd.DataFrame(y_train).to_csv(output_dir / "y_train.csv", index=False)
pd.DataFrame(y_val).to_csv(output_dir / "y_val.csv", index=False)
pd.DataFrame(y_test).to_csv(output_dir / "y_test.csv", index=False)

# ── Save Preprocessor to artifacts/ ───────────────────────────────────────────
# IMPORTANT: preprocessor.pkl must be saved here so that Rivan (train_models.py)
# and Nathan (cluster_models.py) use the exact same pipeline
artifacts_dir = BASE_DIR / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")

print("Preprocessing complete.")
print(f"Training set size (after SMOTE): {X_train.shape}")
print(f"Validation set size:             {X_val.shape}")
print(f"Test set size:                   {X_test.shape}")
print(f"Preprocessor saved to:           {artifacts_dir / 'preprocessor.pkl'}")
