#  Imports
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import shap
import warnings; warnings.filterwarnings('ignore')

#  Load in Preprocessed Data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

X_val   = pd.read_csv(DATA_DIR / "X_val.csv")
y_val   = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

#  Tune Decision Tree with GridSearchCV

dt_params = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'criterion': ['gini']
}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     dt_params, cv=3, scoring='f1_weighted', n_jobs=-1)
dt_gs.fit(X_train, y_train)
print("Best DT params:", dt_gs.best_params_)
best_dt = dt_gs.best_estimator_

#  Saving Decision Tree model as an artifact

joblib.dump(best_dt, 'artifacts/model_dt.pkl')
print("Saved model_dt.pkl", dt_gs.best_params_)

#  Tune Random Forest with RandomizedSearchCV

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                            rf_params, n_iter=10, cv=3,
                            scoring='f1_weighted', random_state=42)
rf_rs.fit(X_train, y_train)
print("Best RF params:", rf_rs.best_params_)
best_rf = rf_rs.best_estimator_

#  Saving the Random Forest model as an artifact

joblib.dump(best_rf, 'artifacts/model_rf.pkl')
print("Saved model_rf.pkl", rf_rs.best_params_)

#  Tune XGBoost with RandomizedSearchCV

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
xgb_rs = RandomizedSearchCV(XGBClassifier(random_state=42,
                             eval_metric='mlogloss', n_jobs=-1),
                             xgb_params, n_iter=10, cv=3,
                             scoring='f1_weighted', random_state=42)
xgb_rs.fit(X_train, y_train_enc)
print("Best XGB params:", xgb_rs.best_params_)
best_xgb = xgb_rs.best_estimator_

#  Saving the XGBoost model as an artifact

joblib.dump(best_xgb, 'artifacts/model_xgb.pkl')
print("Saved model_xgb.pkl", xgb_rs.best_params_)

#  Comparison Summary Table 

models = {
    'Decision Tree': (best_dt, X_test, y_test),
    'Random Forest': (best_rf, X_test, y_test),
    'XGBoost':       (best_xgb, X_test, y_test_enc),
}
results = []
for name, (model, Xt, yt) in models.items():
    preds = model.predict(Xt)
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(yt, preds), 4),
        'F1 (weighted)': round(f1_score(yt, preds, average='weighted'), 4)
    })

results_df = pd.DataFrame(results).sort_values('F1 (weighted)', ascending=False)

#  Determining Best Model

best_model_name = results_df.iloc[0]['Model']
best_model_obj  = models[best_model_name][0]
print(f"Best model: {best_model_name}")

#  Prediction csv

dt_preds  = best_dt.predict(X_test)
rf_preds  = best_rf.predict(X_test)
xgb_preds = le.inverse_transform(best_xgb.predict(X_test))

preds_df = pd.DataFrame({
    'y_true':     y_test.values,
    'y_pred_dt':  dt_preds,
    'y_pred_rf':  rf_preds,
    'y_pred_xgb': xgb_preds
})
preds_df.to_csv('artifacts/predictions.csv', index=False)

#  Feature Importance from all 3 Models

fi_df = pd.DataFrame({
    'feature':       X_train.columns,
    'importance_dt':  best_dt.feature_importances_,
    'importance_rf':  best_rf.feature_importances_,
    'importance_xgb': best_xgb.feature_importances_
}).sort_values('importance_rf', ascending=False)

fi_df.to_csv('artifacts/feature_importance.csv', index=False)
print(fi_df.head(10))

top10 = fi_df.head(10)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top10['feature'], top10['importance_rf'], color='steelblue')
ax.invert_yaxis()
ax.set_title('Top 10 features — Random Forest')
plt.tight_layout()
plt.savefig('reports/figures/feature_importance_rf.png', dpi=150)
plt.show()

for name, preds in [('DT', dt_preds), ('RF', rf_preds), ('XGB', xgb_preds)]:
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')

#Handoff notes for Shap Analysis. The best model was dtermined to be XGBoost with an accuracy rating of 91.35%