# Explainable Machine Learning for Music Popularity Prediction

A Temporal-Drift and Genre-Specificity Analysis

**COM748 Masters Research Project — Ulster University**
Thaneshowar Joshi | Supervisor: Kosmas Kosmopoulos

---

## Overview

This project builds a machine learning system that predicts Spotify track popularity from audio features and metadata, benchmarks four ensemble models, and directly tests two questions rarely addressed in prior work:

- **Does a model trained on older music stay accurate on newer releases?** (temporal drift)
- **Does a single global model suffice, or do genre-specific models perform better?** (genre specificity)

Explainability (via SHAP) is treated as a core objective, not an afterthought — every prediction the demonstration platform makes is accompanied by an explanation of which features drove it.

## Objectives

1. Benchmark four ensemble tree-based models — Random Forest, XGBoost, LightGBM, and CatBoost — under identical, tuning-neutral conditions.
2. Apply SHAP explainability, validated by an independent ablation study.
3. Test genre-specific models against a single global model.
4. Evaluate temporal-drift robustness by training on an earlier era of releases and testing on a disjoint later era.
5. Deliver a working, self-contained Streamlit demonstration platform.

## Data Sources

| Dataset | Size | Contains | Used for |
|---|---|---|---|
| [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) | 114,000 tracks, 114 genres | Popularity scores + audio features | Main modelling, SHAP, genre-specific experiments |
| [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) | 1.2 million tracks | Release dates + audio features (no popularity) | Temporal-drift experiment |

Only 4,729 tracks (5.3%) overlap between the two datasets by Spotify track ID — too sparse to merge for the main modelling, but sufficient for a focused temporal-drift experiment.

## Methodology

1. **Data cleaning & EDA** — missing-value handling, genre-balance check, popularity-distribution analysis.
2. **Leakage-safe feature engineering** — train/test split performed *before* any target-dependent feature (e.g. artist-popularity history) is computed.
3. **Model training** — four models trained under identical default hyperparameters (`n_estimators=200`, `random_state=42`) for a fair, tuning-neutral comparison.
4. **Explainability** — SHAP TreeExplainer on the best-performing model, cross-checked with a feature-ablation study.
5. **Genre-specific evaluation** — per-genre models compared against the global model on matched test subsets.
6. **Temporal-drift evaluation** — trained on 2010–2016 releases, tested on a disjoint 2017–2020 holdout, compared against a same-era baseline.
7. **Demonstration platform** — Streamlit app for interactive prediction + SHAP explanation.

Every stochastic operation uses a fixed random seed (`42`) for full reproducibility.

## Key Results

**Model comparison (held-out test set)**

| Model | RMSE | R² | Accuracy | F1 | AUC |
|---|---|---|---|---|---|
| **Random Forest** | **13.15** | **0.652** | **87.5%** | **0.819** | **0.900** |
| XGBoost | 14.06 | 0.602 | 86.6% | 0.817 | 0.891 |
| LightGBM | 14.49 | 0.577 | 85.5% | 0.798 | 0.888 |
| CatBoost | 14.57 | 0.573 | 85.6% | 0.800 | 0.879 |

- **Random Forest outperforms all three gradient-boosting models** — contrary to prior literature claiming gradient boosting is the consistent state-of-the-art for this task.
- **SHAP + ablation**: artist-popularity history is the dominant feature, but removing it only drops R² from 0.652 to 0.525 — audio features alone retain 80% of the model's explanatory power.
- **Genre-specific models underperform the global model in 6 of 8 genres tested** — each genre model trains on only 800 tracks vs. 91,199 for the global model.
- **Temporal drift is substantial**: a model trained on 2010–2016 releases achieves R²=0.240 on 2017–2020 tracks, vs. R²=0.616 for a same-era baseline — a drift gap of 0.376, indicating static models need periodic retraining for real-world deployment.

## Demonstration Platform

A Streamlit app allows interactive lookup of any track in the held-out dataset, showing:
- Predicted vs. actual popularity
- A SHAP waterfall plot explaining the prediction

The platform is fully self-contained (no live external API calls), following a Spotify Developer Mode policy change in February 2026 that restricted new third-party applications' access to bulk metadata endpoints.

## Repository Contents



## Getting Started

1. Clone the repository and open `Music_Popularity.ipynb` in Google Colab or Jupyter.
2. Download the two datasets from Kaggle (links above) into a `data/` folder, or configure the Kaggle API to fetch them directly (see the first cells of the notebook).
3. Run the notebook cells in order — the pipeline is fully reproducible given the fixed random seed (`42`).
4. To run the demonstration platform locally:
```bash
   pip install streamlit shap pandas scikit-learn joblib
   streamlit run app.py
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · Streamlit · matplotlib

## Limitations

- Temporal-drift experiment uses a comparatively small sample (3,311 tracks), limited by the overlap between the two source datasets.
- Hyperparameters were held at defaults across all models to keep the comparison tuning-neutral; a fully-tuned comparison might change the ranking.
- Popularity is influenced by marketing, playlist placement, and social-network effects not captured by any feature used here.

## Future Work

- Hyperparameter-tuned re-comparison of all four models.
- A hierarchical or multi-task genre-aware architecture.
- A larger, longer-horizon temporal-drift replication.
- Incorporating network and marketing-effect features.

## Author

**Thaneshowar Joshi** — MSc Computing, Ulster University
Supervisor: Kosmas Kosmopoulos
