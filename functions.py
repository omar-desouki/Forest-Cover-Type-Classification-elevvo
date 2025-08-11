import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import shutil
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_random_forest(
    X, y, test_size=0.2, random_state=42, grid_search=False, balanced=None
):
    """
    Train a Random Forest model with optional grid search and evaluate performance.

    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series
        Target variable (Cover_Type)
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    grid_search : bool, default=False
        Whether to perform grid search for hyperparameter tuning
    balanced : str or None, default=None
        Class weight balancing ('balanced' or None)

    Returns:
    --------
    dict : Dictionary containing model, predictions, and evaluation metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if grid_search:
        # Define parameter grid for grid search
        # param_grid = {
        #     "n_estimators": [100, 200, 300],
        #     "max_depth": [None, 10, 20, 30],
        #     "min_samples_split": [2, 5, 10],
        #     "min_samples_leaf": [1, 2, 4],
        #     "max_features": ["sqrt", "log2", None],
        #     "bootstrap": [True, False],
        # }

        param_grid = {
            "n_estimators": [200, 300],  # Reduced from [100, 200, 300]
            "max_depth": [10, 20, None],  # Reduced from [None, 10, 20, 30]
            "min_samples_split": [5, 10],  # Reduced from [2, 5, 10]
            "min_samples_leaf": [2, 4],  # Reduced from [1, 2, 4]
            "max_features": ["sqrt", "log2"],  # Reduced from ["sqrt", "log2", None]
            "bootstrap": [True, False],
        }

        # Create Random Forest model
        rf = RandomForestClassifier(
            random_state=random_state, class_weight=balanced, verbose=0
        )

        # Perform grid search
        print(f"Performing grid search using {os.cpu_count()} CPU cores...")
        grid_search_cv = GridSearchCV(
            rf, param_grid, cv=5, scoring="f1_macro", n_jobs=os.cpu_count(), verbose=3
        )
        grid_search_cv.fit(X_train, y_train)

        # Use best model
        model = grid_search_cv.best_estimator_
        print(f"Best parameters: {grid_search_cv.best_params_}")
        print(f"Best cross-validation score: {grid_search_cv.best_score_:.4f}")

    else:
        # Train simple Random Forest
        # model = RandomForestClassifier(
        #     random_state=random_state, n_estimators=100, class_weight=balanced
        # )
        # after grid search this was the best model
        model = RandomForestClassifier(
            bootstrap=False,
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=300,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # Print evaluation results for training
    print("\n" + "=" * 50)
    print("RANDOM FOREST EVALUATION RESULTS (train set)")
    print("=" * 50)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average="macro")
    train_recall = recall_score(y_train, y_train_pred, average="macro")
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    print(f"Training Accuracy:  {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Training Recall:    {train_recall:.4f}")
    print(f"Training F1-Score:  {train_f1:.4f}")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Print evaluation results
    print("\n" + "=" * 50)
    print("RANDOM FOREST EVALUATION RESULTS (test set)")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y),
    )
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted Cover Type")
    plt.ylabel("Actual Cover Type")
    plt.tight_layout()
    plt.show()

    # Feature importance (Random Forest specific)
    if hasattr(model, "feature_importances_"):
        print("\nTop 15 Feature Importances:")
        feature_names = (
            X.columns
            if hasattr(X, "columns")
            else [f"Feature_{i}" for i in range(X.shape[1])]
        )
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(feature_importance.head(15))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x="importance", y="feature")
        plt.title("Top 15 Feature Importances - Random Forest")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

    return {
        "model": model,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": (
            feature_importance if hasattr(model, "feature_importances_") else None
        ),
    }
