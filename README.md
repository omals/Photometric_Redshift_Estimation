# On the Performance of Machine Learning Models and Hybrid Ensembles with Neural Embeddings for Photometric Redshift Estimation

**Leveraging Machine Learning and Ensemble Techniques for Reliable Photometric Redshift Estimation**

This repository contains the complete implementation of the research work:

> **Omal S., Dr. Madhu S. Nair**

The study systematically evaluates classical machine learning models, deep neural networks, and hybrid ensemble architectures for large-scale photometric redshift (photo-z) estimation using SDSS data .

---

# 📌 Project Overview

Photometric redshift (photo-z) estimation enables scalable redshift prediction using multi-band photometric observations, avoiding expensive spectroscopic measurements.

This project:

* Uses **36 optimized photometric + morphological features**
* Evaluates classical ML, ensemble, and deep learning models
* Proposes **hybrid neural-ensemble architectures**
* Benchmarks performance across two redshift regimes

---

# 📊 Dataset

## Source

* Sloan Digital Sky Survey (SDSS DR18)

## Dataset Statistics

* Original dataset: ~4.25M objects
* Cleaned dataset: ~3.7M objects 
* Train / Validation / Test split: **70 / 15 / 15**

## Redshift Ranges

* **D1**: 0 ≤ z ≤ 2 (Low redshift)
* **D2**: 0 ≤ z ≤ 8 (Full range)

## Selected Features (36)

Includes:

* ugriz magnitudes
* Petrosian radii & fluxes
* PSF magnitudes
* Axis ratios (expAB)
* Model magnitudes
* Derived color indices: (u−g, g−r, r−i, i−z)

Feature selection performed using Pearson correlation with redshift .

---

# 🧠 Implemented Models

## 1️⃣ Classical Regression

* Linear Regression
* Ridge
* Lasso
* Elastic Net
* KNN
* SVR
* Gaussian Process Regression

## 2️⃣ Tree-Based Ensembles

* Decision Tree
* Random Forest
* XGBoost

## 3️⃣ Deep Learning Models

### Neural Network (NN)

Architecture:

```
128 → 64 → 32 → 1
```

* ReLU
* LayerNorm
* Dropout (0.2)
* Adam (lr=0.001)
* MSE Loss 

### Fully Connected Network (FCN)

```
100 → 65 → 35 → 1
```

---

## 4️⃣ Hybrid Ensemble Models (Best Performing)

* NN + Random Forest
* NN + XGBoost
* FCN + Random Forest
* FCN + XGBoost

Hybrid models combine:

* Neural embeddings (representation learning)
* Ensemble regression (variance reduction & stability)

Performance improvements confirmed experimentally .

---

# 🏆 Key Results

## D1 (0 ≤ z ≤ 2)

| Model                  | RMSE       | R²         |
| ---------------------- | ---------- | ---------- |
| Random Forest          | 0.1615     | 0.8603     |
| XGBoost                | 0.1539     | 0.8734     |
| Neural Network         | 0.1592     | 0.8645     |
| **NN + Random Forest** | **0.1521** | **0.8763** |

## D2 (0 ≤ z ≤ 8)

| Model                  | RMSE       | R²         |
| ---------------------- | ---------- | ---------- |
| Random Forest          | 0.3747     | 0.7089     |
| XGBoost                | 0.3591     | 0.7324     |
| Neural Network         | 0.3596     | 0.7315     |
| **NN + Random Forest** | **0.3566** | **0.7361** |

Hybrid approaches consistently outperform standalone models .

---

# 🏗 Repository Structure

```
photoz-hybrid-ensemble/
│
├── Data/
│   ├── values content loaded
│   ├── correlation.py
│   ├── distribution.py
│   └── preprocessing.ipynb
│
├── Model/
│   ├── FCN_xgboost.py
│   ├── NN_xgboost.py
│   ├── decision_tree.py
│   ├── gaussian_process_regression.ipynb
│   ├── knn.py
│   ├── linear_models.ipynb
│   ├── randomforest.py
│   ├── svr.py
│   └── xgboost.py
│
├── Output/
│   │
│   ├── Images/
│   │   ├── knn_d1.jpeg
│   │   ├── knn_d2.jpeg
│   │   ├── randomforest_d1.jpeg
│   │   ├── randomforest_d2.jpeg
│   │   ├── redshift_dist.jpeg
│   │   ├── svr_d1.jpeg
│   │   ├── svr_d2.jpeg
│   │   ├── xgboost_d1.jpeg
│   │   └── xgboost_d2.jpeg
│   │
│   └── values/
│       ├── Hybrid_ensemble model results.txt
│       ├── knn_d1.txt
│       ├── knn_d2.txt
│       ├── randomforest_d1.txt
│       ├── randomforest_d2.txt
│       ├── svr_d1.txt
│       ├── svr_d2.txt
│       ├── xgboostd1.txt
│       └── xgboostd2.txt
│
├── LICENSE
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

```bash
git clone https://github.com/your-username/photoz-hybrid-ensemble.git
cd photoz-hybrid-ensemble
pip install -r requirements.txt
```

## Requirements

* Python 3.9+
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* PyTorch
* Matplotlib

---

# 🚀 Usage

## Train Neural Network

```bash
python train.py 
```

## Train Hybrid Model

```bash
python train.py 
```

## 5-Fold Cross Validation

```bash
python cross_validation.py 
```

---

# 🖥 Computational Setup

Experiments were conducted on:

* Dual AMD EPYC (64 cores)
* Dual NVIDIA A100 (80GB)
* CUDA 12.9 

---

# 🔬 Scientific Contributions

* Comprehensive comparison of regression techniques
* Demonstrates strong non-linearity in photo-z problem
* Shows superiority of hybrid neural + ensemble models
* Scalable framework for large sky surveys

---

# 🌠 Applicability to Future Surveys

This framework is suitable for next-generation missions:

* Vera C. Rubin Observatory (LSST)
* Euclid
* Nancy Grace Roman Space Telescope

These missions require accurate, scalable photometric redshift estimation .

---

# 📚 Related Literature

* Review of photo-z techniques 
* Early SDSS photo-z implementation 

---

# 🔮 Future Work

* Uncertainty estimation (photo-z PDFs)
* Outlier detection
* Domain adaptation
* Explainability (SHAP analysis on embeddings)
* Probabilistic calibration

---

# 👨‍💻 Author

**Omals**
Department of Computer Science
CUSAT

Guided by **Dr. Madhu S. Nair**

---

# 📌 Summary

This repository demonstrates that:

> **Hybrid neural embedding + ensemble regression provides a robust, scalable, and high-accuracy solution for large-scale photometric redshift estimation.**

---
