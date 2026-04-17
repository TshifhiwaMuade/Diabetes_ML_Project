from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_model_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ── Create Output Folders ──────────────────────────────────────────────────────
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Dataset ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ── Verify Column Names ────────────────────────────────────────────────────────
# NOTE: "age" is lowercase in the dataset - confirm before running
print("Dataset columns:", df.columns.tolist())

# ── Select Clustering Features ─────────────────────────────────────────────────
# NOTE: Update "Age" to "age" if dataset uses lowercase column names
selected_features = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c"
]

# ── Validate Selected Features Exist in Dataset ────────────────────────────────
missing_cols = [col for col in selected_features if col not in df.columns]
if missing_cols:
    raise KeyError(
        f"The following selected features were not found in the dataset: "
        f"{missing_cols}. Check column name casing."
    )

# ── Keep Only Selected Features ────────────────────────────────────────────────
X = df[selected_features].copy()

# ── Scale Features ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Apply K-Means Clustering ───────────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# ── Apply PCA for Visualisation ────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ── Derive Cluster Name Mapping Dynamically ────────────────────────────────────
# Cluster labels from K-Means are arbitrary and can shuffle between runs.
# Names are derived from actual cluster center values to ensure consistency.
centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers = pd.DataFrame(centers_original_scale, columns=selected_features)
cluster_centers["cluster"] = range(kmeans.n_clusters)

higher_is_healthier = [
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "hdl_cholesterol"
]

lower_is_healthier = [
    "age",
    "alcohol_consumption_per_week",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "ldl_cholesterol",
    "triglycerides",
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c"
]

glucose_risk_features = [
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c"
]

cluster_centers["overall_risk_score"] = (
    cluster_centers[lower_is_healthier].mean(axis=1) -
    cluster_centers[higher_is_healthier].mean(axis=1)
)
cluster_centers["glucose_risk_score"] = cluster_centers[glucose_risk_features].mean(axis=1)

lower_risk_cluster = cluster_centers.loc[
    cluster_centers["overall_risk_score"].idxmin(), "cluster"
]

remaining_clusters = cluster_centers.loc[
    cluster_centers["cluster"] != lower_risk_cluster
].copy()

glucose_elevated_cluster = remaining_clusters.loc[
    remaining_clusters["glucose_risk_score"].idxmax(), "cluster"
]

cardiometabolic_cluster = remaining_clusters.loc[
    remaining_clusters["cluster"] != glucose_elevated_cluster, "cluster"
].iloc[0]

cluster_names = {
    int(lower_risk_cluster): "active lower-risk profile",
    int(glucose_elevated_cluster): "glucose-elevated high-risk profile",
    int(cardiometabolic_cluster): "cardiometabolic risk profile"
}

# ── Evaluate Clustering ────────────────────────────────────────────────────────
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# ── Save Cluster Labels with Names ────────────────────────────────────────────
cluster_labels = pd.DataFrame({"cluster": clusters})
cluster_labels["cluster_name"] = cluster_labels["cluster"].map(cluster_names)
cluster_labels.to_csv(ARTIFACTS_DIR / "cluster_labels.csv", index=False)

# ── Create and Save Cluster Profiles ──────────────────────────────────────────
df_profiles = df[selected_features].copy()
df_profiles["cluster"] = clusters
df_profiles["cluster_name"] = df_profiles["cluster"].map(cluster_names)
cluster_profiles = df_profiles.groupby(["cluster", "cluster_name"]).mean()
cluster_profiles.to_csv(ARTIFACTS_DIR / "cluster_profiles.csv")

print("\nCluster Profiles:")
print(cluster_profiles)

# ── Save K-Means Model and Scaler to Artifacts ────────────────────────────────
# IMPORTANT: These must be saved so that:
# - Tshifhiwa can load kmeans_model.pkl for SHAP analysis
# - Nasisipho can load both files for the dashboard to assign new patients to clusters
joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans_model.pkl")
joblib.dump(scaler, ARTIFACTS_DIR / "cluster_scaler.pkl")

print(f"\nK-Means model saved to:  {ARTIFACTS_DIR / 'kmeans_model.pkl'}")
print(f"Cluster scaler saved to: {ARTIFACTS_DIR / 'cluster_scaler.pkl'}")

# ── Create PCA Cluster Plot ────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.colorbar(scatter, label="cluster")
plt.title("Patient Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(FIGURES_DIR / "cluster_pca.png")
plt.close()

print(f"\nPCA plot saved to: {FIGURES_DIR / 'cluster_pca.png'}")
print("\nClustering complete.")
