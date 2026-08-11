import os

# ============================================================
# MATPLOTLIB BACKEND
# ============================================================

# Important when running Matplotlib through Flask on macOS.
# It prevents GUI/window errors.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# DATASET PATH
# ============================================================

CSV_PATH = r"Policypulse.csv"


# ============================================================
# PLOT DIRECTORY
# ============================================================

PLOT_DIR = os.path.join(
    "static",
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

sns.set(
    style="whitegrid"
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_rows",
    200
)


# ============================================================
# SAVE PLOT FUNCTION
# ============================================================

def show(
    title="",
    filename="plot.png"
):

    if title:
        plt.title(title)

    plt.tight_layout()

    filepath = os.path.join(
        PLOT_DIR,
        filename
    )

    plt.savefig(
        filepath,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close("all")

    return filename


# ============================================================
# RUN EDA
# ============================================================

def run_eda():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    data = pd.read_csv(
        CSV_PATH
    )


    # ========================================================
    # BASIC CLEANING
    # ========================================================

    # Remove unnecessary unnamed column if present

    if "Unnamed: 0" in data.columns:

        data = data.drop(
            columns=["Unnamed: 0"]
        )


    # ========================================================
    # 1. DATA LOADED
    # ========================================================

    print("=" * 80)

    print("1. DATA LOADED")

    print("=" * 80)

    print(
        "Shape of data:",
        data.shape
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        data.head()
    )


    # List to store generated plots

    plot_files = []


    # ========================================================
    # 2. BASIC INFO
    # ========================================================

    print("\n" + "=" * 80)

    print("2. BASIC INFO")

    print("=" * 80)


    print(
        "\nColumn Data Types:"
    )

    print(
        data.dtypes
    )


    print(
        "\nNumeric Description:"
    )

    print(
        data.describe()
    )


    print(
        "\nCategorical Description:"
    )

    print(
        data.describe(
            include="object"
        )
    )


    # ========================================================
    # 3. MISSING VALUES
    # ========================================================

    print("\n" + "=" * 80)

    print("3. MISSING VALUES")

    print("=" * 80)


    missing = data.isnull().sum()


    missing_pct = (
        missing / len(data)
    ) * 100


    missing_df = pd.DataFrame({

        "missing_count":
            missing,

        "missing_pct":
            missing_pct

    })


    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        by="missing_pct",
        ascending=False
    )


    if missing_df.empty:

        print(
            "No missing values found."
        )

    else:

        print(
            missing_df
        )


        plt.figure(
            figsize=(10, 5)
        )


        sns.barplot(
            x=missing_df.index,
            y=missing_df[
                "missing_count"
            ]
        )


        plt.xticks(
            rotation=45,
            ha="right"
        )


        plt.xlabel(
            "Columns"
        )


        plt.ylabel(
            "Missing Count"
        )


        filename = show(
            title="Missing Values by Column",
            filename="missing_values.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 4. DUPLICATE ROWS
    # ========================================================

    print("\n" + "=" * 80)

    print("4. DUPLICATE ROWS")

    print("=" * 80)


    duplicate_count = int(
        data.duplicated().sum()
    )


    print(
        "Duplicate rows:",
        duplicate_count
    )


    # ========================================================
    # 5. TARGET VARIABLE - RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("5. TARGET VARIABLE - RESPONSE")

    print("=" * 80)


    if "Response" in data.columns:

        response_counts = (
            data["Response"]
            .value_counts()
            .sort_index()
        )


        print(
            "\nResponse Counts:"
        )

        print(
            response_counts
        )


        response_percentage = (
            data["Response"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )


        print(
            "\nResponse Percentage:"
        )

        print(
            response_percentage
        )


        # Response count plot

        plt.figure(
            figsize=(7, 5)
        )


        sns.countplot(
            data=data,
            x="Response"
        )


        plt.xlabel(
            "Response (0 = No, 1 = Yes)"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Customer Response Distribution",
            filename="response_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 6. AGE DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)

    print("6. AGE DISTRIBUTION")

    print("=" * 80)


    if "Age" in data.columns:

        print(
            data["Age"].describe()
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.histplot(
            data["Age"],
            bins=30,
            kde=True
        )


        plt.xlabel(
            "Age"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Age Distribution",
            filename="age_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 7. ANNUAL PREMIUM DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)

    print("7. ANNUAL PREMIUM DISTRIBUTION")

    print("=" * 80)


    if "Annual_Premium" in data.columns:

        print(
            data["Annual_Premium"].describe()
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.histplot(
            data["Annual_Premium"],
            bins=40,
            kde=True
        )


        plt.xlabel(
            "Annual Premium"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Annual Premium Distribution",
            filename="annual_premium_distribution.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 8. OUTLIER DETECTION
    # ========================================================

    print("\n" + "=" * 80)

    print("8. OUTLIER DETECTION")

    print("=" * 80)


    box_cols = [

        "Age",

        "Annual_Premium",

        "Vintage",

        "Region_Code",

        "Policy_Sales_Channel"

    ]


    box_cols = [

        col

        for col in box_cols

        if col in data.columns

    ]


    for index, col in enumerate(
        box_cols
    ):

        plt.figure(
            figsize=(10, 4)
        )


        sns.boxplot(
            x=data[col]
        )


        plt.xlabel(
            col
        )


        filename = show(
            title=f"Box Plot - {col}",
            filename=(
                f"boxplot_{index + 1}_{col}.png"
            )
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 9. CORRELATION ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)

    print("9. CORRELATION ANALYSIS")

    print("=" * 80)


    numeric_data = data.select_dtypes(
        include=[np.number]
    )


    corr = numeric_data.corr()


    print(
        "\nCorrelation Matrix:"
    )


    print(
        np.round(
            corr,
            decimals=2
        )
    )


    plt.figure(
        figsize=(12, 9)
    )


    sns.heatmap(
        np.round(
            corr,
            decimals=2
        ),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )


    filename = show(
        title="Correlation Heatmap",
        filename="correlation_heatmap.png"
    )


    plot_files.append(
        filename
    )


    # ========================================================
    # 10. GENDER VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("10. GENDER VS RESPONSE")

    print("=" * 80)


    if (
        "Gender" in data.columns
        and
        "Response" in data.columns
    ):


        gender_response = pd.crosstab(
            data["Gender"],
            data["Response"]
        )


        print(
            gender_response
        )


        plt.figure(
            figsize=(7, 5)
        )


        sns.countplot(
            data=data,
            x="Gender",
            hue="Response"
        )


        plt.xlabel(
            "Gender"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Gender vs Response",
            filename="gender_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 11. VEHICLE DAMAGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("11. VEHICLE DAMAGE VS RESPONSE")

    print("=" * 80)


    if (
        "Vehicle_Damage" in data.columns
        and
        "Response" in data.columns
    ):


        damage_response = pd.crosstab(
            data["Vehicle_Damage"],
            data["Response"]
        )


        print(
            damage_response
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.countplot(
            data=data,
            x="Vehicle_Damage",
            hue="Response"
        )


        plt.xlabel(
            "Vehicle Damage"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Vehicle Damage vs Response",
            filename="vehicle_damage_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 12. PREVIOUSLY INSURED VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("12. PREVIOUSLY INSURED VS RESPONSE")

    print("=" * 80)


    if (
        "Previously_Insured" in data.columns
        and
        "Response" in data.columns
    ):


        insured_response = pd.crosstab(
            data["Previously_Insured"],
            data["Response"]
        )


        print(
            insured_response
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.countplot(
            data=data,
            x="Previously_Insured",
            hue="Response"
        )


        plt.xlabel(
            "Previously Insured (0 = No, 1 = Yes)"
        )


        plt.ylabel(
            "Number of Customers"
        )


        filename = show(
            title="Previously Insured vs Response",
            filename="previously_insured_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 13. VEHICLE AGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("13. VEHICLE AGE VS RESPONSE")

    print("=" * 80)


    if (
        "Vehicle_Age" in data.columns
        and
        "Response" in data.columns
    ):


        vehicle_age_response = pd.crosstab(
            data["Vehicle_Age"],
            data["Response"]
        )


        print(
            vehicle_age_response
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.countplot(
            data=data,
            x="Vehicle_Age",
            hue="Response"
        )


        plt.xlabel(
            "Vehicle Age"
        )


        plt.ylabel(
            "Number of Customers"
        )


        plt.xticks(
            rotation=20
        )


        filename = show(
            title="Vehicle Age vs Response",
            filename="vehicle_age_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 14. AGE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("14. AGE VS RESPONSE")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Response" in data.columns
    ):


        plt.figure(
            figsize=(8, 5)
        )


        sns.boxplot(
            data=data,
            x="Response",
            y="Age"
        )


        plt.xlabel(
            "Response"
        )


        plt.ylabel(
            "Age"
        )


        filename = show(
            title="Age vs Response",
            filename="age_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 15. ANNUAL PREMIUM VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("15. ANNUAL PREMIUM VS RESPONSE")

    print("=" * 80)


    if (
        "Annual_Premium" in data.columns
        and
        "Response" in data.columns
    ):


        # Sample only for visualization

        premium_plot = data.sample(
            min(
                5000,
                len(data)
            ),
            random_state=42
        )


        plt.figure(
            figsize=(8, 5)
        )


        sns.boxplot(
            data=premium_plot,
            x="Response",
            y="Annual_Premium"
        )


        plt.xlabel(
            "Response"
        )


        plt.ylabel(
            "Annual Premium"
        )


        filename = show(
            title="Annual Premium vs Response",
            filename="annual_premium_vs_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 16. REGION CODE VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("16. REGION CODE VS RESPONSE")

    print("=" * 80)


    if (
        "Region_Code" in data.columns
        and
        "Response" in data.columns
    ):


        region_response = (

            data.groupby(
                "Region_Code"
            )["Response"]

            .mean()

            .sort_values(
                ascending=False
            )

            .head(15)

        )


        print(
            "\nTop Regions by Response Rate:"
        )


        print(
            region_response
        )


        plt.figure(
            figsize=(10, 6)
        )


        sns.barplot(
            x=region_response.index.astype(str),
            y=region_response.values
        )


        plt.xlabel(
            "Region Code"
        )


        plt.ylabel(
            "Response Rate"
        )


        plt.xticks(
            rotation=45
        )


        filename = show(
            title="Top Regions by Response Rate",
            filename="region_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 17. POLICY SALES CHANNEL VS RESPONSE
    # ========================================================

    print("\n" + "=" * 80)

    print("17. POLICY SALES CHANNEL VS RESPONSE")

    print("=" * 80)


    if (
        "Policy_Sales_Channel" in data.columns
        and
        "Response" in data.columns
    ):


        channel_response = (

            data.groupby(
                "Policy_Sales_Channel"
            )["Response"]

            .mean()

            .sort_values(
                ascending=False
            )

            .head(15)

        )


        print(
            "\nTop Policy Sales Channels by Response Rate:"
        )


        print(
            channel_response
        )


        plt.figure(
            figsize=(10, 6)
        )


        sns.barplot(
            x=channel_response.index.astype(str),
            y=channel_response.values
        )


        plt.xlabel(
            "Policy Sales Channel"
        )


        plt.ylabel(
            "Response Rate"
        )


        plt.xticks(
            rotation=45
        )


        filename = show(
            title="Top Policy Sales Channels by Response Rate",
            filename="sales_channel_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 18. AGE VS ANNUAL PREMIUM
    # ========================================================

    print("\n" + "=" * 80)

    print("18. AGE VS ANNUAL PREMIUM")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Annual_Premium" in data.columns
    ):


        scatter_data = data.sample(
            min(
                5000,
                len(data)
            ),
            random_state=42
        )


        scatter_data = scatter_data[
            [
                "Age",
                "Annual_Premium"
            ]
        ].dropna()


        # Remove non-finite values

        scatter_data = scatter_data[
            np.isfinite(
                scatter_data
            ).all(axis=1)
        ]


        plt.figure(
            figsize=(8, 5)
        )


        sns.scatterplot(
            data=scatter_data,
            x="Age",
            y="Annual_Premium",
            alpha=0.4
        )


        plt.xlabel(
            "Age"
        )


        plt.ylabel(
            "Annual Premium"
        )


        filename = show(
            title="Age vs Annual Premium",
            filename="age_vs_premium.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 19. RESPONSE RATE BY AGE GROUP
    # ========================================================

    print("\n" + "=" * 80)

    print("19. RESPONSE RATE BY AGE GROUP")

    print("=" * 80)


    if (
        "Age" in data.columns
        and
        "Response" in data.columns
    ):


        age_bins = [
            0,
            25,
            35,
            45,
            55,
            65,
            100
        ]


        age_labels = [
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66+"
        ]


        age_groups = pd.cut(
            data["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )


        age_response = (

            data.groupby(
                age_groups,
                observed=False
            )["Response"]

            .mean()

            * 100

        )


        print(
            "\nResponse Rate by Age Group:"
        )


        print(
            age_response
        )


        plt.figure(
            figsize=(9, 5)
        )


        sns.barplot(
            x=age_response.index,
            y=age_response.values
        )


        plt.xlabel(
            "Age Group"
        )


        plt.ylabel(
            "Response Rate (%)"
        )


        filename = show(
            title="Response Rate by Age Group",
            filename="age_group_response.png"
        )


        plot_files.append(
            filename
        )


    # ========================================================
    # 20. PAIRPLOT
    # ========================================================

    print("\n" + "=" * 80)

    print("20. PAIRPLOT")

    print("=" * 80)


    pair_cols = [

        "Age",

        "Annual_Premium",

        "Vintage",

        "Response"

    ]


    pair_cols = [

        col

        for col in pair_cols

        if col in data.columns

    ]


    if len(pair_cols) >= 2:

        pair_data = data[
            pair_cols
        ].sample(

            min(
                2000,
                len(data)
            ),

            random_state=42

        ).dropna()


        pair_data = pair_data[
            np.isfinite(
                pair_data
            ).all(axis=1)
        ]


        pair_grid = sns.pairplot(
            pair_data,
            hue="Response" if "Response"
            in pair_data.columns else None,
            diag_kind="hist"
        )


        pair_grid.savefig(

            os.path.join(
                PLOT_DIR,
                "pairplot.png"
            ),

            dpi=120,

            bbox_inches="tight"

        )


        plt.close("all")


        plot_files.append(
            "pairplot.png"
        )


    # ========================================================
    # RESPONSE RATE
    # ========================================================

    response_rate = None


    if "Response" in data.columns:

        response_rate = round(
            data["Response"].mean() * 100,
            2
        )


    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "rows": int(
            data.shape[0]
        ),

        "columns": int(
            data.shape[1]
        ),

        "column_names": list(
            data.columns
        ),

        "duplicate_rows":
            duplicate_count,

        "response_rate":
            response_rate,

        "missing_values": {

            column: int(count)

            for column, count
            in data.isnull().sum().items()

        },

        "preview":
            data.head(10).to_dict(
                orient="records"
            ),

        "plots":
            plot_files

    }


    return results


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    results = run_eda()


    print("\n" + "=" * 80)

    print(
        "POLICYPULSE EDA COMPLETED"
    )

    print("=" * 80)


    print(
        "\nRows:",
        results["rows"]
    )


    print(
        "Columns:",
        results["columns"]
    )


    print(
        "Duplicate Rows:",
        results["duplicate_rows"]
    )


    print(
        "Response Rate:",
        results["response_rate"],
        "%"
    )


    print(
        "\nPlots saved in:",
        PLOT_DIR
    )