# Diabetes Risk Segmentation & Decision Support System
**BC Analytics | MLG382 Guided Project 2026 | CRISP-DM Framework**

---

## Project Overview
This project builds a data-driven decision support system for early identification and management of diabetes risk. Using the Diabetes and Lifestyle Dataset (97,297 records, 31 features), the system:
- Classifies patients into diabetes risk categories (No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational)
- Identifies key lifestyle drivers of risk using SHAP analysis
- Groups patients into meaningful lifestyle-based segments using K-Means clustering
- Delivers actionable recommendations through an interactive DASH web application

---

## Project Structure
```
diabetes_project/
|________data/
|    |______raw/                  # Original unmodified dataset
|    |______processed/            # Train/test splits after preprocessing
|
|________src/
|    |______prepare_data.py       # EDA and data cleaning (Member 1)
|    |______preprocess_data.py    # Encoding, scaling, SMOTE (Member 2)
|    |______train_models.py       # DT, RF, XGBoost classifiers (Member 3)
|    |______cluster_models.py     # K-Means clustering (Member 4)
|    |______shap_analysis.py      # SHAP values and visualisations (Member 5)
|    |______web_app.py            # DASH web application (Member 6)
|
|________artifacts/               # Saved models (.pkl), predictions, SHAP values
|________notebooks/               # EDA and modelling Jupyter notebooks
|________reports/                 # Technical report (CRISP-DM structured)
|________requirements.txt         # Project dependencies
|________ReadMe.md
```

---

## Team Members & Branches

| Member | Role | Branch |
|--------|------|--------|
| Nicholas Sunnasy (601353) | Data Lead | `nicholas-data-lead` |
| Stephen van der Merwe (601789) | Feature Engineer | `stephen-feature-engineer` |
| Rivan Matitz (601530) | Classifier | `rivan-classifier` |
| Nathan Labuschagne (602113) | Clustering Specialist | `nathan-clustering` |
| Tshifhiwa Muade (576941) | Key Driver Analyst | `tshifhiwa-shap-analysis` |
| Nasisipho Mbana (602139) | Deployment & Reporting Lead | `nasisipho-deployment` |

---

## Getting Started

### Step 1 — Clone the Repository
```bash
git clone https://github.com/TshifhiwaMuade/Diabetes_ML_Project.git
cd Diabetes_ML_Project
```

### Step 2 — Set Up Your Environment (Anaconda)
1. Open **Anaconda Navigator**
2. Click **Environments** in the left sidebar
3. Click **Create** at the bottom
4. Name it `diabetes_project`, select **Python 3.11**, click **Create**
5. Click the **play button** next to the environment
6. Select **Open Terminal**
7. Navigate to the project folder:
```bash
cd "path/to/diabetes_project"
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create Your Branch
Each member must work on their own branch. Replace `your-branch-name` with your branch from the table above:
```bash
git checkout -b your-branch-name
```

### Step 5 — Launch JupyterLab
```bash
jupyter lab
```

---

## Workflow & Handoff Dependencies

```
Nicholas (EDA & Cleaning)
        ↓
Stephen (Encoding, Scaling, SMOTE)
        ↓               ↓
Rivan (Classification)  Nathan (Clustering)
        ↓               ↓
      Tshifhiwa (SHAP Analysis)
              ↓
        Nasisipho (Dashboard & Deployment)
```

> ⚠️ Members 3 and 4 cannot begin until Member 1 and 2 hand off a clean, model-ready dataset.
> ⚠️ Member 5 cannot begin until both the best classifier and cluster labels are finalised.

---

## Pushing Your Work to GitHub
After completing your work on your branch:
```bash
git add .
git commit -m "meaningful description of what you did"
git push origin your-branch-name
```
Then open a **Pull Request** on GitHub to merge your branch into `main`.

---

## Deliverables
- Technical Report (CRISP-DM structured, max 2 pages)
- GitHub Repository with meaningful commit history
- Interactive DASH Web App (deployed on Render)
- Video Demo

---

## Links
- **GitHub Repository:** https://github.com/TshifhiwaMuade/Diabetes_ML_Project
- **Web Application:** *(link to be added after deployment)*
