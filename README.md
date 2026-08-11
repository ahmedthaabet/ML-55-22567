## Classification Project

**File Reference:** `classification_project_ahmed_thabet_55-22567_3.ipynb`

### Step 1: Exploratory Data Analysis (EDA)
The initial step of this project performs an EDA on the `car_crash_train.csv` training dataset to understand the underlying structure, relationships, and potential anomalies within the data.

**Objectives:**
*   **Data Overview:** Inspect the dataset shape, column data types, and initial rows.
*   **Summary Statistics:** Analyze the central tendency and dispersion of numerical features.
*   **Data Integrity Checks:** Identify any missing (null) values or duplicate records.
*   **Target Variable Analysis:** Examine the distribution of the `Severity` target variable to detect potential class imbalance.
*   **Feature Visualizations:** Plot distributions for numerical features and frequency counts for categorical features.
*   **Correlation Analysis:** Visualize the linear correlations between numerical features to check for multicollinearity.

---

### Dataset Information
*   **Shape:** The dataset contains 4000 rows and 19 columns.
*   **Data Cleaning:** The `Distraction Level` column was corrected by filling missing values with the string category 'None'.
*   **Data Integrity:** After cleaning, there are 0 missing values and 0 duplicate rows detected.

---

### Target Variable Distribution ('Severity')
An analysis of the target variable reveals a class imbalance heavily weighted toward minor injuries:
*   **Minor Injury:** 2756 counts (68.90%).
*   **Severe Injury:** 1038 counts (25.95%).
*   **Fatal:** 206 counts (5.15%).

---

### Visualizations
*   **Target Variable:** A countplot was generated using the `Set2` and `viridis` palettes to display the distribution of crash severity. 
*   **Numerical Features:** Histograms utilizing 20 bins were plotted to map the distributions of all numerical features within the dataset.

# Clustering Project

**Deadline for submission:** 17/3/2026 @ 11:59 PM

## Objective
The main objective of this project is to apply clustering techniques to real datasets. 

The project aims to address and justify the following questions:
* Which clustering approach is decided for each dataset?
* How do KMeans, Hierarchical, and DBScan clustering compare?
* How are clustering hyperparameters tuned to achieve the best cluster assignment?
* What is the effect of different distance functions on the calculated clusters?
* How is the performance of different clustering techniques evaluated on different datasets?
* How can the outputs be visualized?
* What is the effect of scaling on the performance of clustering techniques?

---

## Setup and Dependencies
The project utilizes the following libraries for data manipulation, machine learning algorithms, and visualization:
* `pandas` and `numpy`
* `matplotlib.pyplot` and `seaborn`
* `scikit-learn` (specifically `DBSCAN`, `KMeans`, `AgglomerativeClustering`, `StandardScaler`, `PCA`, `make_blobs`, and `silhouette_score`)
* `scipy.cluster.hierarchy`

---

## Multi Blob Dataset
A synthetic dataset was generated using the `make_blobs` function to represent a scenario known to be best clustered into 6 clusters. 
* **Centers:** The data was generated around 6 specific coordinates: (-3, -3), (0, 0), (5, 2.5), (-1, 4), (4, 6), and (9, 7).
* **Samples:** The dataset contains varied sample sizes per cluster: 100, 150, 300, 400, 300, and 200 observations.
* **Cluster Standard Deviations:** 1.3, 0.6, 1.2, 1.7, 0.9, and 1.7.

A custom helper function, `display_cluster()`, is provided to plot the data in two dimensions, highlight the assigned clusters, and mark the centroids with an "x".

---

## Clustering Analysis

### KMeans Clustering
The dataset was evaluated using the KMeans algorithm for different values of K (from 2 to 10) to determine the optimal number of clusters by analyzing the Silhouette Score and Distortion:

*   **K=2:** Silhouette Score: 0.4752 | Distortion: 16716.6192
*   **K=3:** Silhouette Score: 0.4336 | Distortion: 12187.2885
*   **K=4:** Silhouette Score: 0.4625 | Distortion: 7812.3220
*   **K=5:** Silhouette Score: 0.4809 | Distortion: 5600.3053
*   **K=6:** Silhouette Score: 0.4862 | Distortion: 4310.5581
*   **K=7:** Silhouette Score: 0.4634 | Distortion: 3811.2118
*   **K=8:** Silhouette Score: 0.4112 | Distortion: 3563.3477
*   **K=9:** Silhouette Score: 0.4497 | Distortion: 3089.3639
*   **K=10:** Silhouette Score: 0.4006 | Distortion: 2854.7798

**Conclusion:** The analysis identifies that the best K based on the Silhouette Score is **6**.

# Machine Learning Lab Tasks

## Overview
This repository contains a collection of Jupyter Notebooks dedicated to various Machine Learning lab tasks. The tasks cover unsupervised learning (Clustering) and supervised learning (Classification and Regression), along with Exploratory Data Analysis (EDA) and hyperparameter tuning techniques.

## Datasets Used
*   **Bank Marketing Dataset**: A dataset related to direct marketing campaigns of a Portuguese banking institution, used for EDA and clustering tasks.
*   **Sonar Data**: Contains response metrics for 60 separate sonar frequencies, used to classify objects as either rocks or mines.
*   **Banknote Authentication Dataset**: Used to predict the authenticity of banknotes based on variance, skewness, curtosis, and entropy.

---

## Lab Tasks Breakdown

### 1. Clustering Algorithms

#### K-Means Clustering
*   **Description**: A partition-based clustering algorithm that divides the data into `k` distinct, non-overlapping clusters by minimizing the variance within each cluster.
*   **Key Parameters Evaluated**:
    *   `n_clusters`: The number of clusters to form as well as the number of centroids to generate.
    *   `init`: Method for initialization (e.g., `k-means++` to speed up convergence).
    *   `max_iter`: Maximum number of iterations of the k-means algorithm for a single run.

#### Agglomerative (Hierarchical) Clustering
*   **Description**: A deterministic, bottom-up hierarchical clustering method that builds a tree (dendrogram) of clusters. 
*   **Key Parameters Evaluated**:
    *   `n_clusters`: The desired number of clusters.
    *   `metric`: Distance metric (e.g., `euclidean`, `manhattan`, `cosine`).
    *   `linkage`: Linkage criteria (`ward`, `complete`, `average`). Note: The `ward` linkage only allows the `euclidean` metric.
    *   `distance_threshold`: Stops merging when the cluster distance exceeds this limit.

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
*   **Description**: Groups closely packed points into dense regions and labels points in low-density areas as noise. 
*   **Key Concepts**: Identification of Core points, Border points, and Noise points.
*   **Key Parameters Evaluated**:
    *   `eps`: The maximum distance between two points to be considered neighbors.
    *   `min_samples`: The minimum number of points required to form a dense region (core point).

### 2. Supervised Learning Algorithms (Classification & Regression)

#### Linear Regression
*   **Objective**: Predict a continuous target variable based on one or more independent variables by fitting a linear equation to observed data.
*   **Methodology**: Evaluating the model's line of best fit and minimizing the residual sum of squares between observed and predicted targets.

#### Logistic Regression
*   **Objective**: Estimate the probability that an instance belongs to a particular class (primarily used for binary or multiclass classification).
*   **Methodology**: Utilizing the logistic (sigmoid) function to map predictions to probabilities between 0 and 1. 
*   **Key Parameters**: Tuning parameters such as `C` (inverse of regularization strength) and selecting the appropriate `solver`.

#### Regularization Techniques (Ridge, Lasso, ElasticNet)
*   **Objective**: Prevent overfitting in regression and classification models by adding a penalty term to the loss function.
*   **Techniques Covered**:
    *   **L1 Regularization (Lasso)**: Adds an absolute value penalty; can shrink some coefficients to zero, effectively performing feature selection.
    *   **L2 Regularization (Ridge)**: Adds a squared magnitude penalty; shrinks coefficients evenly to reduce model complexity without eliminating features.
    *   **ElasticNet**: A hybrid approach combining both L1 and L2 penalties.

#### Naive Bayes
*   **Objective**: Perform probabilistic classification based on Bayes' Theorem, with the "naive" assumption of conditional independence between every pair of features.
*   **Variants Evaluated**: Gaussian Naive Bayes (typically for continuous data) and/or Multinomial Naive Bayes (for discrete counts).

#### K-Nearest Neighbors (KNN)
*   **Objective**: Detect whether a sonar response is bouncing off a rock or a mine.
*   **Methodology**: 
    *   Implementing a Scikit-Learn `Pipeline` combining a `StandardScaler` and a `KNeighborsClassifier`.
    *   Performing hyperparameter tuning using `GridSearchCV` to find the optimal `k` value (`n_neighbors`).

#### Decision Trees and Random Forests
*   **Objective**: Classify banknotes as genuine or forged.
*   **Methodology**:
    *   Training a `DecisionTreeClassifier` and tuning parameters like `criterion`, `max_depth`, and `max_leaf_nodes` using `GridSearchCV`.
    *   Training a `RandomForestClassifier` and tuning parameters like `n_estimators`, `max_depth`, and `max_features`.
    *   Evaluating and comparing both models using the Confusion Matrix and Classification Report metrics.

---

## Evaluation Metrics
Across the notebooks, the following metrics and tools are utilized to evaluate model performance:
*   **Clustering**: Silhouette Score, and Inertia (specifically for evaluating K-Means).
*   **Classification**: Accuracy, F1-Score, Precision, Recall, Confusion Matrix, and Classification Report.
*   **Regression**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared ($R^2$).

## Dependencies
To run the notebooks, you will need the following Python libraries:
*   `numpy`
*   `pandas`
*   `matplotlib.pyplot`
*   `seaborn`
*   `scikit-learn`
