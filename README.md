# Comprehensive Machine Learning Pipeline: Heart Disease Prediction

This project implements a full machine learning workflow, from data preprocessing to model optimization, to predict the presence of heart disease using the **Heart Disease UCI Dataset**.

All analytical steps are contained within a **single, comprehensive notebook**, making the entire pipeline easy to follow and reproduce.

# 📊 Project Goals and Deliverables

The core objective was to build a reliable classifier for heart disease. This involved several key stages:

* **Data Preparation:** The pipeline loads, cleans, scales, and explores the raw data.
* **Dimensionality Reduction:** **Principal Component Analysis (PCA)** is applied to reduce the feature count, and the results (variance plot) are documented.
* **Modeling:** Multiple supervised models (Logistic Regression, Random Forest, SVM) and one unsupervised model (K-Means) are trained and evaluated, producing **performance metrics** (Accuracy, F1-Score, AUC).
* **Optimization:** The best model is fine-tuned using GridSearchCV to create a **hyperparameter optimized model**.
* **Model Export:** The final optimized model is saved in the required **`.pkl` format** for future deployment.

# 🛠️ Tools and Libraries Used

* **Programming Language:** Python
* **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
