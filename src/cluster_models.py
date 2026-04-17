from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_model_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# create output folders if they do not exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# load dataset
df = pd.read_csv(DATA_PATH)

# select clustering features
selected_features = [
    "Age",
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


# keep only selected features
X = df[selected_features].copy()

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# apply k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# apply pca for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# create cluster name mapping
cluster_names = {
    0: "active lower-risk profile",
    1: "glucose-elevated high-risk profile",
    2: "cardiometabolic risk profile"
}

# evaluate clustering
score = silhouette_score(X_scaled, clusters)
print("silhouette score:")
print(score)

# save cluster labels with names
cluster_labels = pd.DataFrame({
    "cluster": clusters
})
cluster_labels["cluster_name"] = cluster_labels["cluster"].map(cluster_names)
cluster_labels.to_csv(ARTIFACTS_DIR / "cluster_labels.csv", index=False)

# create cluster profiles
df_profiles = df[selected_features].copy()
df_profiles["cluster"] = clusters
df_profiles["cluster_name"] = df_profiles["cluster"].map(cluster_names)
cluster_profiles = df_profiles.groupby(["cluster", "cluster_name"]).mean()

# save profiles
cluster_profiles.to_csv(ARTIFACTS_DIR / "cluster_profiles.csv")

# print profiles
print("\ncluster profiles:")
print(cluster_profiles)

# create pca plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.colorbar(scatter, label="cluster")
plt.title("patient clusters (pca projection)")
plt.xlabel("pca 1")
plt.ylabel("pca 2")
plt.savefig(FIGURES_DIR / "cluster_pca.png")
plt.close()
