# Forest Cover Type Classification

A machine learning project to predict forest cover types using cartographic variables derived from the US Geological Survey (USGS) and US Forest Service (USFS) data.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)

## 🌲 Overview

This project implements a Random Forest classifier to predict forest cover types based on cartographic variables. The model classifies seven different cover types found in four wilderness areas located in the Roosevelt National Forest of northern Colorado.

### Cover Types:

1. **Spruce/Fir**
2. **Lodgepole Pine**
3. **Ponderosa Pine**
4. **Cottonwood/Willow**
5. **Aspen**
6. **Douglas-fir**
7. **Krummholz**

## 📊 Dataset

The dataset contains 581,012 observations with 54 attributes including:

- **Cartographic Variables**: Elevation, aspect, slope, distances to water/roads/fire points
- **Soil Types**: 40 binary soil type variables
- **Wilderness Areas**: 4 binary wilderness area variables

**Source**: [UCI ML Forest Cover Type Dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

## 🔧 Features

### Topographic Features:

- **Elevation** (meters)
- **Aspect** (azimuth, 0-360 degrees)
- **Slope** (degrees)
- **Horizontal_Distance_To_Hydrology** (meters)
- **Vertical_Distance_To_Hydrology** (meters)
- **Horizontal_Distance_To_Roadways** (meters)
- **Horizontal_Distance_To_Fire_Points** (meters)
- **Hillshade_9am**, **Hillshade_Noon**, **Hillshade_3pm** (0-255 index)

### Categorical Features:

- **Wilderness_Area** (4 binary columns, 0/1)
- **Soil_Type** (40 binary columns, 0/1)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages (see requirements below)

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/omar-desouki/Forest-Cover-Type-Classification-elevvo.git
   cd Forest-Cover-Type-Classification-elevvo
   ```

2. **Install required packages**:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```

3. **Run the notebook**:

   ```bash
   jupyter notebook main.ipynb
   ```

## 💻 Usage

### Quick Start

1. **Open the Jupyter notebook** (`main.ipynb`)
2. **Run all cells** to execute the complete pipeline:
   - Data download and preprocessing
   - Exploratory data analysis
   - Model training and evaluation

### Key Functions

The `functions.py` module contains the main training function:

```python
from functions import train_random_forest

# Train Random Forest with default parameters
results = train_random_forest(X, y)

# Train with grid search for hyperparameter tuning
results = train_random_forest(X, y, grid_search=True)

# Train with balanced class weights
results = train_random_forest(X, y, balanced='balanced')
```

## 📁 Project Structure

```
Forest-Cover-Type-Classification-elevvo/
├── main.ipynb              # Main Jupyter notebook
├── functions.py            # Utility functions for model training
├── data/                   # Dataset directory
│   └── covtype.csv         # Forest cover type dataset
├── __pycache__/            # Python cache files
└── README.md               # Project documentation
```

## 🔬 Methodology

### 1. Data Exploration & Analysis

- **Data Overview**: 581,012 samples, 54 features, 7 target classes
- **Distribution Analysis**: Target class imbalance identified
- **Correlation Analysis**: Feature relationships examined
- **Visualization**: Histograms, box plots, and correlation heatmaps

### 2. Data Preprocessing

- **Missing Values**: No missing values found
- **Duplicates**: Duplicate checking performed
- **Feature Engineering**: No additional feature engineering required (data pre-processed)

### 3. Model Training

- **Algorithm**: Random Forest Classifier
- **Train/Test Split**: 80/20 stratified split
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Class Balancing**: Optional balanced class weights

### 4. Model Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-Score (macro-averaged)
- **Visualization**: Confusion matrix and feature importance plots
- **Cross-validation**: 5-fold CV during grid search

## 📈 Results

The Random Forest model was evaluated in three different configurations, showing strong performance across all variants:

### Model 1: Default Hyperparameters

```python
train_random_forest(X, y, grid_search=False, balanced=None)
```

**Configuration:**

- **n_estimators**: 100 (default)
- **max_depth**: None (unlimited)
- **min_samples_split**: 2 (default)
- **min_samples_leaf**: 1 (default)
- **max_features**: 'sqrt' (default)
- **bootstrap**: True (default)

**Performance Metrics:**
- **Test Accuracy**: 95.33%
- **Precision**: 94.63%
- **Recall**: 90.53%
- **Macro F1-Score**: 92.41%
- **Training Time**: Fast (baseline)

### Model 2: Grid Search Optimized

```python
train_random_forest(X, y, grid_search=True, balanced=None)
```

**Best Configuration Found:**

- **n_estimators**: 300
- **max_depth**: None (unlimited)
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **max_features**: 'sqrt'
- **bootstrap**: False

**Performance Metrics:**
- **Test Accuracy**: 95.64%
- **Precision**: 94.80%
- **Recall**: 91.06%
- **Macro F1-Score**: 92.79%
- **Cross-validation Score**: ~93.1%
- **Training Time**: Longer (due to grid search)

### Model 3: Grid Search + Balanced Classes

```python
train_random_forest(X, y, grid_search=False, balanced='balanced')
```

**Configuration:**

- **n_estimators**: 300 (from best grid search)
- **max_depth**: None
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **max_features**: 'sqrt'
- **bootstrap**: False
- **class_weight**: 'balanced'

**Performance Metrics:**
- **Test Accuracy**: 95.64%
- **Precision**: 94.80%
- **Recall**: 91.06%
- **Macro F1-Score**: 92.79%
- **Better performance on minority classes**
- **Reduced bias towards majority classes**

### Performance Comparison Summary:

| Model Variant | Test Accuracy | Precision | Recall | Macro F1-Score |
|---------------|---------------|-----------|--------|----------------|
| Default Parameters | 95.33% | 94.63% | 90.53% | 92.41% |
| Grid Search Optimized | 95.64% | 94.80% | 91.06% | 92.79% |
| Balanced Classes | 95.64% | 94.80% | 91.06% | 92.79% |

### Top Important Features (Consistent across all models):

1. **Elevation** - Most discriminative feature
2. **Horizontal Distance to Roadways** - Infrastructure proximity
3. **Horizontal Distance to Fire Points** - Fire risk indicator
4. **Various Soil Types** - Geological characteristics
5. **Hillshade measurements** - Topographic lighting conditions

## 🛠️ Model Features

### Grid Search Parameters:

```python
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}
```

### Evaluation Outputs:

- Detailed classification report
- Confusion matrix visualization
- Feature importance ranking
- Performance metrics comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 Notes

- The dataset is automatically downloaded using Kaggle API
- Model training can be computationally intensive with grid search
- Feature importance analysis helps understand model decisions
- Class imbalance is addressed through stratified splitting and optional balancing

## 🔗 References

- [UCI ML Repository - Forest Cover Type Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

---

**Author**: Omar Desouki
**Project Type**: Machine Learning Classification
**Framework**: Scikit-learn, Pandas, Matplotlib, Seaborn
