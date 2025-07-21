# Iris Flower Classification with Machine Learning

## Introduction

This project demonstrates a complete machine learning workflow for classifying Iris flowers into three species: Setosa, Versicolor, and Virginica. The classification is based on the physical measurements of the flower's sepals and petals. This is a classic and widely used dataset in the field of machine learning, making it an excellent project for practising fundamental classification techniques.

The project covers all stages of the machine learning lifecycle, from data exploration and feature engineering to model training, hyperparameter tuning, and in-depth evaluation. The final trained model is also serialized for easy deployment and use in other applications.

---

## Dataset

The project utilizes the well-known Iris dataset.

* **Source:** The data is loaded directly from a public GitHub repository.
* **Features:** The dataset includes four primary features:
    * `SepalLengthCm`
    * `SepalWidthCm`
    * `PetalLengthCm`
    * `PetalWidthCm`
* **Target Variable:** The target variable is `Species`, which contains three distinct classes:
    * `Iris-setosa`
    * `Iris-versicolor`
    * `Iris-virginica`
* **Data Quality:** The dataset is clean and balanced, with 50 samples for each of the three species, totaling 150 samples.

---

## Project Workflow

The project is structured in a clear, step-by-step manner:

1.  **Data Loading and Exploration:**
    * The dataset is loaded into a pandas DataFrame.
    * Initial exploratory data analysis (EDA) is performed using `.info()`, `.describe()`, and visualizations like box plots and a correlation heatmap to understand the data's structure and feature relationships.

2.  **Data Preprocessing and Feature Engineering:**
    * The unnecessary `Id` column is removed.
    * The categorical `Species` labels are converted into numerical format using `LabelEncoder`.
    * Two new features, `SepalArea` and `PetalArea`, are engineered to potentially capture more complex patterns in the data.
    * A `pairplot` is generated to visualize the relationships between the features and their separability by species.

3.  **Model Training and Selection:**
    * Three different classification algorithms are evaluated: **Random Forest**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)**.
    * 5-fold cross-validation is used to get a robust estimate of each model's performance and to select the best candidate for further tuning.

4.  **Hyperparameter Tuning:**
    * The best-performing model, Random Forest, is selected for hyperparameter optimization.
    * `GridSearchCV` is employed to systematically search for the optimal combination of hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).

5.  **Model Evaluation:**
    * The final, tuned model is evaluated on a held-out test set.
    * Performance is measured using a comprehensive set of metrics:
        * **Accuracy Score**
        * **Classification Report** (Precision, Recall, F1-Score)
        * **Confusion Matrix**
        * **Feature Importance Plot**
        * **Multi-class ROC Curve** with AUC scores.

6.  **Model Deployment:**
    * The trained `RandomForestClassifier` and the `LabelEncoder` are saved to disk as `.pkl` files using `joblib`. This allows for easy loading and use in other applications without needing to retrain the model.

---

## Results

* The model comparison showed that all three algorithms performed exceptionally well, with Random Forest, SVM, and KNN all achieving cross-validation scores around **95-96%**.
* After hyperparameter tuning with `GridSearchCV`, the optimized Random Forest model achieved a perfect **100% accuracy** on the test set.
* The **Feature Importance** analysis revealed that `PetalArea` and `PetalLengthCm` were the most significant predictors for classifying the Iris species, which aligns with the high correlation observed during EDA.

---


## Libraries Used

* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning models, preprocessing, and evaluation.
* **Seaborn & Matplotlib:** For data visualization.
* **Joblib:** For saving and loading the trained model.
