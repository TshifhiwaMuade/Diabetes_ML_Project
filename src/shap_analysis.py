from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
DATA_DIR = BASE_DIR / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Models and Data ───────────────────────────────────────────────────────
print("Loading models and data...")

# Load best classifier (XGBoost)
xgb_model = joblib.load(ARTIFACTS_DIR / "model_xgb.pkl")

# Load preprocessor from Stephen
preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")

# Load K-Means model and scaler from Nathan
kmeans_model = joblib.load(ARTIFACTS_DIR / "kmeans_model.pkl")
cluster_scaler = joblib.load(ARTIFACTS_DIR / "cluster_scaler.pkl")

# Load cluster labels and profiles
cluster_labels = pd.read_csv(ARTIFACTS_DIR / "cluster_labels.csv")
cluster_profiles = pd.read_csv(ARTIFACTS_DIR / "cluster_profiles.csv")

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print(f"X_test shape: {X_test.shape}")
print(f"Cluster labels shape: {cluster_labels.shape}")

# ── Feature Names ──────────────────────────────────────────────────────────────
# Reconstruct feature names after OneHotEncoding
numerical_cols = [
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "Age"
]

categorical_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status"
]

try:
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + cat_feature_names
    print(f"Total features after encoding: {len(feature_names)}")
except Exception as e:
    print(f"Could not retrieve feature names: {e}")
    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

# ── SECTION 1 — SHAP on XGBoost Classifier ────────────────────────────────────
print("\nRunning SHAP analysis on XGBoost classifier...")

# Use TreeExplainer for XGBoost - fastest and most accurate for tree models
explainer_xgb = shap.TreeExplainer(xgb_model)

# Use a sample of 2000 rows to keep computation manageable
sample_size = 2000
X_test_sample = X_test.iloc[:sample_size]
y_test_sample = y_test.iloc[:sample_size]

shap_values_xgb = explainer_xgb.shap_values(X_test_sample)

print("SHAP values computed for XGBoost.")
print(f"SHAP values shape: {np.array(shap_values_xgb).shape}")

# ── Global Feature Importance — Bar Plot ───────────────────────────────────────
print("\nGenerating global bar plot...")

plt.figure()
shap.summary_plot(
    shap_values_xgb,
    X_test_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("Global Feature Importance — XGBoost (Bar)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_bar.png", bbox_inches="tight")
plt.close()
print("Saved shap_bar.png")

# ── Global Feature Importance — Beeswarm Plot ─────────────────────────────────
print("\nGenerating beeswarm plot...")

plt.figure()
shap.summary_plot(
    shap_values_xgb,
    X_test_sample,
    feature_names=feature_names,
    show=False
)
plt.title("Global Feature Importance — XGBoost (Beeswarm)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_beeswarm.png", bbox_inches="tight")
plt.close()
print("Saved shap_beeswarm.png")

# ── Local Explanation — Waterfall Plot (Single Patient) ───────────────────────
print("\nGenerating waterfall plot for a single patient...")

# Multiclass model - select class index to explain
# 0=Gestational, 1=No Diabetes, 2=Pre-Diabetes, 3=Type 1, 4=Type 2
class_index = 4  # Explaining Type 2 prediction

explanation = shap.Explanation(
    values=shap_values_xgb[0, :, class_index],
    base_values=explainer_xgb.expected_value[class_index],
    data=X_test_sample.iloc[0].values,
    feature_names=feature_names
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title("Local Explanation — Patient 0, Type 2 Class (Waterfall)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_waterfall.png", bbox_inches="tight")
plt.close()
print("Saved shap_waterfall.png")

# ── Save SHAP Values to CSV ───────────────────────────────────────────────────
print("\nSaving SHAP values to CSV...")

if isinstance(shap_values_xgb, np.ndarray) and shap_values_xgb.ndim == 3:
    # Multiclass — average absolute SHAP values across all classes
    mean_shap = np.mean(np.abs(shap_values_xgb), axis=2)
else:
    mean_shap = np.abs(shap_values_xgb)

shap_df = pd.DataFrame(mean_shap, columns=feature_names)
shap_df.to_csv(ARTIFACTS_DIR / "shap_values.csv", index=False)
print("Saved shap_values.csv")

# ── SECTION 2 — SHAP on K-Means Clusters ──────────────────────────────────────
print("\nRunning SHAP analysis on K-Means clusters...")

clustering_features = [
    "Age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week",
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol",
    "triglycerides", "glucose_fasting", "glucose_postprandial",
    "insulin_level", "hba1c"
]

df_model = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_model_dataset.csv")
X_cluster = df_model[clustering_features].copy()
X_cluster_scaled = cluster_scaler.transform(X_cluster)

# Use KernelExplainer for K-Means
# NOTE: KernelExplainer is model-agnostic but slower — use a small background sample
background = shap.kmeans(X_cluster_scaled, 10)
cluster_predict_fn = lambda x: kmeans_model.predict(x).reshape(-1, 1).astype(float)

explainer_cluster = shap.KernelExplainer(cluster_predict_fn, background)

# Use a small sample for K-Means SHAP — KernelExplainer is computationally expensive
cluster_sample = X_cluster_scaled[:300]
shap_values_cluster = explainer_cluster.shap_values(cluster_sample)

print("SHAP values computed for K-Means clusters.")

# ── Cluster SHAP Bar Plot ──────────────────────────────────────────────────────
print("\nGenerating cluster SHAP bar plot...")

plt.figure()
shap.summary_plot(
    shap_values_cluster,
    cluster_sample,
    feature_names=clustering_features,
    plot_type="bar",
    show=False
)
plt.title("Cluster Feature Importance — K-Means (Bar)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_cluster_bar.png", bbox_inches="tight")
plt.close()
print("Saved shap_cluster_bar.png")

# ── Save Cluster SHAP Values to CSV ───────────────────────────────────────────
print("\nSaving cluster SHAP values to CSV...")

shap_cluster_arr = np.array(shap_values_cluster)

# Squeeze from (300, 18, 1) to (300, 18)
if shap_cluster_arr.ndim == 3:
    shap_cluster_arr = shap_cluster_arr[:, :, 0]

shap_cluster_df = pd.DataFrame(
    np.abs(shap_cluster_arr),
    columns=clustering_features
)
shap_cluster_df.to_csv(ARTIFACTS_DIR / "shap_cluster_values.csv", index=False)
print("Saved shap_cluster_values.csv")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SHAP Analysis Complete.")
print("="*60)
print("\nFiles saved to reports/figures/:")
print("  shap_bar.png")
print("  shap_beeswarm.png")
print("  shap_waterfall.png")
print("  shap_cluster_bar.png")
print("\nFiles saved to artifacts/:")
print("  shap_values.csv")
print("  shap_cluster_values.csv")