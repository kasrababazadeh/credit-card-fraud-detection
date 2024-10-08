# Credit Card Fraud Detection

This repository contains a machine learning project focused on detecting fraudulent transactions using supervised learning models like **Logistic Regression** and **SVM**. The dataset is highly imbalanced, with fraud cases accounting for only 0.172% of all transactions, so **SMOTE** is used to handle class imbalance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)

## Overview

This project aims to build machine learning models to detect fraudulent credit card transactions. It includes data preprocessing, exploratory data analysis (EDA), feature importance analysis, and the implementation of two main machine learning models (Logistic Regression and SVM) for classification.

## Dataset

- **Source**: The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mojtabanafez/rayan-homework1).
- **Details**: The dataset contains 284,807 transactions over two days, with 492 fraud cases (0.172%).

## Technologies Used
- Python 3.8+
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`
  - `imbalanced-learn` (for SMOTE)
  - `numpy`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset:
    - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mojtabanafez/rayan-homework1) and place it in the project directory.

## Usage

1. **Run the Jupyter notebook**:
    ```bash
    jupyter notebook credit-card-fraud-detection.ipynb
    ```

2. Follow the steps in the notebook to:
    - Perform Exploratory Data Analysis (EDA).
    - Preprocess the data (handling missing values, scaling, etc.).
    - Train models using **Logistic Regression** and **SVM**.
    - Apply **SMOTE** to handle class imbalance.

## Modeling

- **Logistic Regression** and **SVM** are used as classification models.
- **SMOTE** is applied to oversample the minority (fraud) class to balance the dataset.
- **Evaluation Metrics**:
    - Precision, Recall, F1-score, and ROC-AUC score are used for evaluating model performance.
    - Confusion Matrix is plotted to visualize the number of True Positives, False Positives, True Negatives, and False Negatives.

## Results

- **Logistic Regression** and **SVM** models show significant improvement in detecting fraud cases after applying SMOTE.
- Classification reports and confusion matrices are provided to evaluate model performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
