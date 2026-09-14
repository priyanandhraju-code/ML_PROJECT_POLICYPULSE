import os
import re
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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

TRAIN_FILE_PATH = "preprocessed_train.csv"
TEST_FILE_PATH = "preprocessed_test.csv"

TARGET_COLUMN = "Response"

RANDOM_STATE = 42

PLOT_DIRECTORY = os.path.join(
    "static",
    "plots"
)


# ============================================================
# TYPE ALIAS
# ============================================================

ModelRunner = Callable[
    [
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series
    ],
    Tuple[Any, Dict[str, Any]]
]


# ============================================================
# LOAD PREPROCESSED TRAINING DATA
# ============================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the preprocessed training dataset.
    """

    if not os.path.exists(TRAIN_FILE_PATH):
        raise FileNotFoundError(
            f"Preprocessed training dataset not found: "
            f"{TRAIN_FILE_PATH}"
        )

    return pd.read_csv(
        TRAIN_FILE_PATH
    )


# ============================================================
# LOAD PREPROCESSED TEST DATA
# ============================================================

def load_testing_data() -> pd.DataFrame:
    """
    Load the preprocessed testing dataset.
    """

    if not os.path.exists(TEST_FILE_PATH):
        raise FileNotFoundError(
            f"Preprocessed testing dataset not found: "
            f"{TEST_FILE_PATH}"
        )

    return pd.read_csv(
        TEST_FILE_PATH
    )


# ============================================================
# SANITIZE FEATURE NAMES
# ============================================================

def sanitize_feature_names(
    feature_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Make feature names safe for XGBoost.

    XGBoost does not allow feature names containing
    [, ] or < characters.
    """

    cleaned_data = feature_data.copy()

    new_names = []
    used_names = set()

    for column_name in cleaned_data.columns:

        name = str(column_name)

        # Replace characters that XGBoost rejects
        name = name.replace(
            "[",
            "_"
        )

        name = name.replace(
            "]",
            "_"
        )

        name = name.replace(
            "<",
            "_"
        )

        name = name.replace(
            ">",
            "_"
        )

        # Replace other special characters
        name = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            name
        )

        # Remove repeated underscores
        name = re.sub(
            r"_+",
            "_",
            name
        )

        # Remove leading/trailing underscores
        name = name.strip(
            "_"
        )

        # Prevent empty feature names
        if not name:
            name = "feature"

        # Make duplicate names unique
        original_name = name
        counter = 1

        while name in used_names:

            name = (
                f"{original_name}_{counter}"
            )

            counter += 1

        used_names.add(
            name
        )

        new_names.append(
            name
        )

    cleaned_data.columns = new_names

    return cleaned_data


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data(
    training_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate training features and target.
    """

    if TARGET_COLUMN not in training_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in training data."
        )

    x_data = training_data.drop(
        columns=[TARGET_COLUMN]
    )

    y_data = training_data[
        TARGET_COLUMN
    ]

    x_data = sanitize_feature_names(
        x_data
    )

    x_data = x_data.astype(
        "float64"
    )

    y_data = y_data.astype(
        "int64"
    )

    return (
        x_data,
        y_data
    )


# ============================================================
# PREPARE TESTING DATA
# ============================================================

def prepare_testing_data(
    testing_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate testing features and target.
    """

    if TARGET_COLUMN not in testing_data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in testing data."
        )

    x_data = testing_data.drop(
        columns=[TARGET_COLUMN]
    )

    y_data = testing_data[
        TARGET_COLUMN
    ]

    x_data = sanitize_feature_names(
        x_data
    )

    x_data = x_data.astype(
        "float64"
    )

    y_data = y_data.astype(
        "int64"
    )

    return (
        x_data,
        y_data
    )


# ============================================================
# VALIDATE FEATURE COLUMNS
# ============================================================

def validate_feature_columns(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame
) -> None:
    """
    Make sure training and testing datasets
    contain the same feature columns.
    """

    if list(training_features.columns) != list(
        testing_features.columns
    ):
        raise ValueError(
            "Training and testing feature columns "
            "do not match."
        )


# ============================================================
# CALCULATE BALANCED SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(
    target_data: pd.Series
) -> pd.Series:
    """
    Calculate balanced sample weights for
    imbalanced binary classification.
    """

    class_counts = target_data.value_counts()

    total_samples = len(
        target_data
    )

    number_of_classes = len(
        class_counts
    )

    class_weights = {}

    for class_value, class_count in class_counts.items():

        class_weights[class_value] = (
            total_samples
            /
            (
                number_of_classes
                *
                class_count
            )
        )

    sample_weights = target_data.map(
        class_weights
    )

    return sample_weights


# ============================================================
# CALCULATE POSITIVE CLASS WEIGHT
# ============================================================

def calculate_scale_pos_weight(
    target_data: pd.Series
) -> float:
    """
    Calculate the negative-to-positive class ratio
    for XGBoost and LightGBM.
    """

    class_counts = target_data.value_counts()

    negative_count = class_counts.get(
        0,
        0
    )

    positive_count = class_counts.get(
        1,
        0
    )

    if positive_count == 0:
        return 1.0

    return (
        negative_count
        /
        positive_count
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model: Any,
    testing_features: pd.DataFrame,
    testing_target: pd.Series
) -> Dict[str, Any]:
    """
    Calculate classification performance metrics.
    """

    predictions = model.predict(
        testing_features
    )

    if not hasattr(
        model,
        "predict_proba"
    ):
        raise ValueError(
            "The selected model does not "
            "support probability prediction."
        )

    probabilities = model.predict_proba(
        testing_features
    )[:, 1]

    accuracy = accuracy_score(
        testing_target,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        testing_target,
        predictions
    )

    precision = precision_score(
        testing_target,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        testing_target,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        testing_target,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        testing_target,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
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
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

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
        training_features,
        training_target
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 2. RANDOM FOREST
# ============================================================

def run_random_forest(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

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
        training_features,
        training_target
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 3. EXTRA TREES
# ============================================================

def run_extra_trees(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

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
        training_features,
        training_target
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 4. GRADIENT BOOSTING
# ============================================================

def run_gradient_boosting(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

    print(
        "\nRunning Gradient Boosting..."
    )

    sample_weights = calculate_sample_weights(
        training_target
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.10,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    model.fit(
        training_features,
        training_target,
        sample_weight=sample_weights
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 5. ADA BOOST
# ============================================================

def run_adaboost(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

    print(
        "\nRunning AdaBoost..."
    )

    sample_weights = calculate_sample_weights(
        training_target
    )

    model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.50,
        random_state=RANDOM_STATE
    )

    model.fit(
        training_features,
        training_target,
        sample_weight=sample_weights
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 6. XG BOOST
# ============================================================

def run_xgboost(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

    print(
        "\nRunning XGBoost..."
    )

    positive_class_weight = (
        calculate_scale_pos_weight(
            training_target
        )
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.10,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="log_loss",
        scale_pos_weight=positive_class_weight,
        n_jobs=-1
    )

    model.fit(
        training_features,
        training_target
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# 7. LIGHT GBM
# ============================================================

def run_lightgbm(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    training_target: pd.Series,
    testing_target: pd.Series
) -> Tuple[Any, Dict[str, Any]]:

    print(
        "\nRunning LightGBM..."
    )

    positive_class_weight = (
        calculate_scale_pos_weight(
            training_target
        )
    )

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.10,
        max_depth=-1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=positive_class_weight
    )

    model.fit(
        training_features,
        training_target
    )

    metrics = evaluate_model(
        model,
        testing_features,
        testing_target
    )

    return (
        model,
        metrics
    )


# ============================================================
# ALGORITHM MAP
# ============================================================

ALGORITHMS: Dict[
    str,
    Tuple[str, ModelRunner]
] = {

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
# CREATE CONFUSION MATRIX
# ============================================================

def create_confusion_matrix_plot(
    result: Dict[str, Any],
    filename: str
) -> str:

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
# CREATE ROC CURVE
# ============================================================

def create_roc_curve_plot(
    result: Dict[str, Any],
    filename: str
) -> str:

    os.makedirs(
        PLOT_DIRECTORY,
        exist_ok=True
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            result["y_test"],
            result["probabilities"]
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
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


# ============================================================
# RUN SELECTED TREE ALGORITHM
# ============================================================

def run_tree_algorithm(
    algorithm_key: str
) -> Dict[str, Any]:

    if algorithm_key not in ALGORITHMS:

        raise ValueError(
            f"Unknown tree algorithm: "
            f"{algorithm_key}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    training_data = load_training_data()

    testing_data = load_testing_data()

    print(
        "\n" + "=" * 80
    )

    print(
        "PolicyPulse - Tree Based Models"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # GET ALGORITHM
    # --------------------------------------------------------

    algorithm_name, model_runner = (
        ALGORITHMS[
            algorithm_key
        ]
    )

    print(
        "\nSelected Algorithm:"
    )

    print(
        algorithm_name
    )

    # --------------------------------------------------------
    # PREPARE TRAINING DATA
    # --------------------------------------------------------

    training_features, training_target = (
        prepare_training_data(
            training_data
        )
    )

    # --------------------------------------------------------
    # PREPARE TESTING DATA
    # --------------------------------------------------------

    testing_features, testing_target = (
        prepare_testing_data(
            testing_data
        )
    )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    validate_feature_columns(
        training_features,
        testing_features
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    print(
        "\nTraining Dataset Shape:"
    )

    print(
        training_data.shape
    )

    print(
        "\nTesting Dataset Shape:"
    )

    print(
        testing_data.shape
    )

    print(
        "\nNumber of Features:"
    )

    print(
        training_features.shape[1]
    )

    print(
        "\nTraining Target Distribution:"
    )

    print(
        training_target.value_counts()
    )

    print(
        "\nTesting Target Distribution:"
    )

    print(
        testing_target.value_counts()
    )

    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    model, metrics = model_runner(
        training_features,
        testing_features,
        training_target,
        testing_target
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        algorithm_name.upper()
    )

    print(
        "=" * 80
    )

    print(
        "Accuracy:",
        metrics["Accuracy"]
    )

    print(
        "Balanced Accuracy:",
        metrics["Balanced Accuracy"]
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
    # RESULT OBJECT
    # --------------------------------------------------------

    result: Dict[str, Any] = {

        "algorithm": algorithm_name,

        "dataset_shape": (
            training_data.shape[0]
            + testing_data.shape[0],
            training_data.shape[1]
        ),

        "training_shape": (
            training_features.shape[0],
            training_features.shape[1]
        ),

        "testing_shape": (
            testing_features.shape[0],
            testing_features.shape[1]
        ),

        "feature_count": (
            training_features.shape[1]
        ),

        "Accuracy": metrics["Accuracy"],

        "Balanced Accuracy": (
            metrics["Balanced Accuracy"]
        ),

        "Precision": metrics["Precision"],

        "Recall": metrics["Recall"],

        "F1 Score": metrics["F1 Score"],

        "ROC AUC": metrics["ROC AUC"],

        "model": model,

        "X_test": testing_features,

        "y_test": testing_target,

        "predictions": metrics["predictions"],

        "probabilities": metrics["probabilities"]
    }

    # --------------------------------------------------------
    # CREATE CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_filename = (
        f"{algorithm_key}"
        "_confusion_matrix.png"
    )

    result["confusion_matrix_plot"] = (
        create_confusion_matrix_plot(
            result,
            confusion_filename
        )
    )

    # --------------------------------------------------------
    # CREATE ROC CURVE
    # --------------------------------------------------------

    roc_filename = (
        f"{algorithm_key}"
        "_roc_curve.png"
    )

    result["roc_curve_plot"] = (
        create_roc_curve_plot(
            result,
            roc_filename
        )
    )

    return result