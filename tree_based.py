import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = "preprocessed_data.csv"

TARGET_COLUMN = "Response"

TEST_SIZE = 0.20

RANDOM_STATE = 42

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    if not os.path.exists(FILE_PATH):

        raise FileNotFoundError(

            f"Preprocessed dataset not found: "
            f"{FILE_PATH}"

        )

    df = pd.read_csv(
        FILE_PATH
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if TARGET_COLUMN not in df.columns:

        raise ValueError(

            f"Target column "
            f"'{TARGET_COLUMN}' not found."

        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(

        y_test,

        predictions

    )


    precision = precision_score(

        y_test,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        predictions,

        zero_division=0

    )


    roc_auc = roc_auc_score(

        y_test,

        probabilities

    )


    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc,

        "predictions": predictions,

        "probabilities": probabilities

    }


# ============================================================
# 1. DECISION TREE
# ============================================================

def run_decision_tree(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Decision Tree..."
    )


    model = DecisionTreeClassifier(

        random_state=RANDOM_STATE,

        class_weight="balanced",

        max_depth=12,

        min_samples_leaf=10

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 2. RANDOM FOREST
# ============================================================

def run_random_forest(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Random Forest..."
    )


    model = RandomForestClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 3. EXTRA TREES
# ============================================================

def run_extra_trees(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Extra Trees..."
    )


    model = ExtraTreesClassifier(

        n_estimators=100,

        random_state=RANDOM_STATE,

        class_weight="balanced",

        n_jobs=-1,

        max_depth=15,

        min_samples_leaf=5

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 4. GRADIENT BOOSTING
# ============================================================

def run_gradient_boosting(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning Gradient Boosting..."
    )


    model = GradientBoostingClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=3,

        random_state=RANDOM_STATE

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 5. ADABOOST
# ============================================================

def run_adaboost(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning AdaBoost..."
    )


    model = AdaBoostClassifier(

        n_estimators=100,

        learning_rate=0.50,

        random_state=RANDOM_STATE

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 6. XGBOOST
# ============================================================

def run_xgboost(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning XGBoost..."
    )


    model = XGBClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=RANDOM_STATE,

        eval_metric="logloss",

        n_jobs=-1

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# 7. LIGHTGBM
# ============================================================

def run_lightgbm(

    X_train,
    X_test,
    y_train,
    y_test

):

    print(
        "\nRunning LightGBM..."
    )


    model = LGBMClassifier(

        n_estimators=100,

        learning_rate=0.10,

        max_depth=-1,

        num_leaves=31,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1

    )


    model.fit(

        X_train,

        y_train

    )


    metrics = evaluate_model(

        model,

        X_test,

        y_test

    )


    return model, metrics


# ============================================================
# ALGORITHM MAP
# ============================================================

ALGORITHMS = {

    "decision_tree": (

        "Decision Tree",

        run_decision_tree

    ),

    "random_forest": (

        "Random Forest",

        run_random_forest

    ),

    "extra_trees": (

        "Extra Trees",

        run_extra_trees

    ),

    "gradient_boosting": (

        "Gradient Boosting",

        run_gradient_boosting

    ),

    "adaboost": (

        "AdaBoost",

        run_adaboost

    ),

    "xgboost": (

        "XGBoost",

        run_xgboost

    ),

    "lightgbm": (

        "LightGBM",

        run_lightgbm

    )

}


# ============================================================
# RUN SELECTED TREE ALGORITHM
# ============================================================

def run_tree_algorithm(
    algorithm
):

    if algorithm not in ALGORITHMS:

        raise ValueError(

            f"Unknown tree algorithm: "
            f"{algorithm}"

        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_preprocessed_data()


    print("\n" + "=" * 80)

    print(
        "POLICYPULSE - V5 TREE BASED MODELS"
    )

    print("=" * 80)


    print(
        "\nSelected Algorithm:"
    )

    print(
        ALGORITHMS[algorithm][0]
    )


    print(
        "\nDataset Shape:"
    )

    print(
        df.shape
    )


    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    X, y = prepare_data(
        df
    )


    print(
        "\nNumber of Features:"
    )

    print(
        X.shape[1]
    )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(

        X,

        y

    )


    print(
        "\nTraining Shape:"
    )

    print(
        X_train.shape
    )


    print(
        "\nTesting Shape:"
    )

    print(
        X_test.shape
    )


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    display_name, function = ALGORITHMS[algorithm]


    model, metrics = function(

        X_train,

        X_test,

        y_train,

        y_test

    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        display_name.upper()
    )

    print(
        "=" * 80
    )


    print(
        "Accuracy:",
        metrics["Accuracy"]
    )

    print(
        "Precision:",
        metrics["Precision"]
    )

    print(
        "Recall:",
        metrics["Recall"]
    )

    print(
        "F1 Score:",
        metrics["F1 Score"]
    )

    print(
        "ROC AUC:",
        metrics["ROC AUC"]
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "algorithm": display_name,

        "dataset_shape": df.shape,

        "feature_count": X.shape[1],

        "training_shape": X_train.shape,

        "testing_shape": X_test.shape,

        "Accuracy": metrics["Accuracy"],

        "Precision": metrics["Precision"],

        "Recall": metrics["Recall"],

        "F1 Score": metrics["F1 Score"],

        "ROC AUC": metrics["ROC AUC"],

        "model": model,

        "X_test": X_test,

        "y_test": y_test,

        "predictions": metrics["predictions"],

        "probabilities": metrics["probabilities"]

    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(

    result,

    filename

):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )


    matrix = confusion_matrix(

        result["y_test"],

        result["predictions"]

    )


    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix

    )


    display.plot()


    plt.title(

        f'{result["algorithm"]} - Confusion Matrix'

    )


    plt.tight_layout()


    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )


    plt.savefig(

        path,

        dpi=150

    )


    plt.close()


    return path


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve_plot(

    result,

    filename

):

    os.makedirs(

        PLOT_DIRECTORY,

        exist_ok=True

    )


    fpr, tpr, _ = roc_curve(

        result["y_test"],

        result["probabilities"]

    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        fpr,

        tpr,

        label=(

            f'ROC AUC = '
            f'{result["ROC AUC"]:.4f}'

        )

    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(

        f'{result["algorithm"]} - ROC Curve'

    )


    plt.legend()


    plt.grid(
        alpha=0.2
    )


    plt.tight_layout()


    path = os.path.join(

        PLOT_DIRECTORY,

        filename

    )


    plt.savefig(

        path,

        dpi=150

    )


    plt.close()


    return path