# Machine Learning Algorithms Repository
Welcome to my Machine Learning Algorithms repository! This project serves as a comprehensive record of my learning journey in machine learning, focusing on understanding and implementing key algorithms alongside their mathematical derivations.

**Purpose**<br>
The aim of this repository is to solidify my knowledge of machine learning algorithms by implementing each from scratch as I learn. My goal is to eventually include all fundamental and advanced algorithms, along with clear explanations of the underlying mathematics, enabling anyone interested to follow along.

**Current Progress**<br>
As I work through each topic, I’ll update the ReadMe to include:

* A list of algorithms completed
* An overview of concepts covered
* Key takeaways from the mathematical derivations and algorithmic insights

**How to Use This Repository**<br>
This repository is designed for learners and enthusiasts who are interested in exploring machine learning algorithms in depth. Each algorithm folder will contain:

* Code Implementation: A Python-based implementation of the algorithm
* Mathematical Derivation: Step-by-step derivations to build intuition
* Documentation: Markdown notes detailing my thought process, approach, and key insights

**Getting Started**<br>
To get started, clone this repository:

```bash
git clone https://github.com/Aniketh007/Machine-Learning.git
```

**Future Plans**<br>
I aim to include more algorithms and concepts as I learn them, gradually expanding the scope and depth of the repository. I will also continue refining the explanations and optimizing the code as I grow in this field.

## Updates

- **Linear Regression**
  - Implemented a Linear Regression model from scratch.
  - Documented the mathematical derivation, including:
    - Cost Function
    - Gradient Descent Algorithm
    - Cross-Validation (5 types)
  - Regularization Techniques:
    - Ridge Regression
    - Lasso Regression
    - ElasticNet Regression
  - **Projects Completed:**
    - Simple Linear Regression:
      - Salary Prediction based on Years of Experience
    - Multiple Regression:
      - Car Price Prediction
      - Life Expectancy WHO
    - Polynomial Regression:
      - Diamond Price Prediction
      - Custom Data Polynomial Regression

        
- **Logistic Regression**
  - Implemented a Logistic Regression model from scratch.
  - Documented the mathematical derivations, including:
    - Sigmoid Activation Function
    - Cost Function
    - Performance Metrics
    - One-vs-Rest (OVR) Strategy
  - Hyperparameter Tuning:
    - GridSearch CV
    - RandomizedSearch CV
  - Logistic Regression on Imbalanced Datasets.
  - **Projects Completed:**
    - Coronary Heart Disease Prediction (next 10 Years)
    - Titanic Survival Prediction
   
      
- **Support Vector Machine**
  - Implemented SVM Model for:
    - Support Vector Classifier(SVC)
    - Support Vector Regressor(SVR)
  - Documented the mathematical derivations, including:
    - Marginal Lines/Planes
      - Soft Margin
      - Hard Margin
    - Cost Function
  - Kernels:
    - Polynomial Kernels
    - RBF Kernels
    - Sigmoid Kernels
  - **Projects Completed:**
    - Human Activity Recognition using Smartphones
    - Insurance Claims Prediction

- **Naive Bayes**
  - Implemented Naive Bayes model from scratch.
  - Documented the mathematical derivations, including:
    - Probability
    - Baye's Theorem
    - Variants of Naive Bayes:
      - Bernoulli Naive Bayes
      - Multinomial Naive Bayes
      - Gaussian Naive Bayes
  - **Projects Completed**:
    - Breast Cancer Diagnosis
- **K Nearest Neighbour**
  - Implemented KNN from scratch.
  - Documented the mathematical derivations, including:
    - Hyperparameter tuning for optimal K value
    - Distance formulas to calculate K nearest neighbour:
      - Euclidean Distance
      - Manhattan Distance
    - Time complexity analysis of KNN (O(N))
    - Optimization of Time Complexity:
      - KD Tree
      - Ball Tree
  - **Project Completed:**:
    - Cancer Prediction
- **Decision Tree**
  - Implemented Decision Tree from scratch.
  - Documented the mathematical derivations, including:
    - Calculation of Information Gain using Entropy.
    - Gini Index for node splitting.
    - Handling categorical and continuous data during tree construction.
    - Pruning techniques:
      - Pre-pruning.
      - Post-pruning to avoid overfitting.
    - Time Complexity Analysis (O(N * log N)).
  - **Project Completed**:  
    - Breast Cancer Prediction.
- **Random Forest**
  - Implemented Random Forest from scratch.
  - Documented the mathematical derivations, including:
    - Bagging technique for creating multiple decision trees with bootstrapped samples.
    - Random feature selection for splitting nodes to reduce correlation between trees.
    - Aggregation of predictions using majority voting for classification.
    - Hyperparameter tuning:
      - Number of trees (`n_trees`).
      - Maximum depth of trees (`max_depth`).
      - Minimum samples required for splits (`min_splits`).
      - Number of features considered for splits (`n_features`).
    - Time Complexity Analysis (O(M * N * log N)):
      - `M` = Number of trees.
      - `N` = Number of samples.
  - **Project Completed**:  
    - Bank Loan Default Prediction.

      
(Will be uploading the notes soon)


Stay tuned for updates!
