# fraud_dashboard.py
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# -------------------------------------------------------------
# 🎨 Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Healthcare Provider Fraud Detection", layout="wide")
st.title("🏥 Healthcare Provider Fraud Detection Dashboard")
st.markdown(
    "Analyze healthcare claims, detect potential provider fraud, and visualize insights interactively."
)


# -------------------------------------------------------------
# 🧠 Streamlit Caching for Performance
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(inpatient_file, outpatient_file, beneficiary_file, train_file):
    inpatient_df = pd.read_csv(inpatient_file)
    outpatient_df = pd.read_csv(outpatient_file)
    beneficiary_df = pd.read_csv(beneficiary_file)
    train_df = pd.read_csv(train_file)
    return inpatient_df, outpatient_df, beneficiary_df, train_df


@st.cache_data(show_spinner="Processing and engineering features...")
def preprocess_and_engineer(
    inpatient_df,
    outpatient_df,
    beneficiary_df,
    train_df,
    is_train=True,
):
    # Convert date columns
    date_columns = [col for col in inpatient_df.columns if "dt" in col.lower()]
    for date_col in date_columns:
        if date_col in inpatient_df.columns:
            inpatient_df[date_col] = pd.to_datetime(inpatient_df[date_col])
        if date_col in outpatient_df.columns:
            outpatient_df[date_col] = pd.to_datetime(outpatient_df[date_col])

    for data in [inpatient_df, outpatient_df, beneficiary_df]:
        data.drop_duplicates(inplace=True)

    claims_df = pd.concat([inpatient_df, outpatient_df], axis=0)
    claims_df = pd.merge(claims_df, beneficiary_df, on=["BeneID"], how="left")

    claims_df["daysAdmitted"] = (
        claims_df["DischargeDt"] - claims_df["AdmissionDt"]
    ).dt.days.fillna(0)
    claims_df["beneAge"] = np.where(
        claims_df["DOD"].isnull(),
        dt.datetime.now().year - pd.to_datetime(claims_df["DOB"]).dt.year,
        pd.to_datetime(claims_df["DOD"]).dt.year
        - pd.to_datetime(claims_df["DOB"]).dt.year,
    )

    # Feature creation
    procedures = [c for c in claims_df.columns if "procedurecode" in c.lower()]
    diagnosis = [c for c in claims_df.columns if "diagnosiscode" in c.lower()]
    chronic_cond = [c for c in claims_df.columns if "chroniccond" in c.lower()]
    all_columns = procedures + diagnosis + chronic_cond

    claims_df["procedureCount"] = claims_df[procedures].count(axis=1)
    claims_df["diagnosisCount"] = claims_df[diagnosis].count(axis=1)
    claims_df[chronic_cond] = claims_df[chronic_cond].replace({2: 0})
    claims_df["chronicCondCount"] = claims_df[chronic_cond].sum(axis=1)

    claims_df["inpatientAnomaly"] = np.where(
        (claims_df["NoOfMonths_PartACov"] == 0)
        & (claims_df["IPAnnualReimbursementAmt"] > 0),
        1,
        0,
    )
    claims_df["outpatientAnomaly"] = np.where(
        (claims_df["NoOfMonths_PartBCov"] == 0)
        & (claims_df["OPAnnualReimbursementAmt"] > 0),
        1,
        0,
    )
    claims_df["noPhysician"] = np.where(
        (claims_df["AttendingPhysician"].isnull())
        & (claims_df["InscClaimAmtReimbursed"] > 0),
        1,
        0,
    )

    cleaned_df = claims_df.drop(all_columns, axis=1, inplace=False)

    provider_summary = (
        cleaned_df.groupby(["Provider"])
        .agg(
            numClaims=("ClaimID", "count"),
            totalClaimReimbersed=("InscClaimAmtReimbursed", "sum"),
            maxClaimReimbursed=("InscClaimAmtReimbursed", "max"),
            meanClaimReimbersed=("InscClaimAmtReimbursed", "mean"),
            ipAnnualReimbursementAmt=("IPAnnualReimbursementAmt", "mean"),
            opAnnualReimbursementAmt=("OPAnnualReimbursementAmt", "mean"),
            totalProcedures=("procedureCount", "sum"),
            totalDiagnosis=("diagnosisCount", "sum"),
            totalchronicCond=("chronicCondCount", "sum"),
            inpatientAnomaly=("inpatientAnomaly", "sum"),
            outpatientAnomaly=("outpatientAnomaly", "sum"),
            noPhysician=("noPhysician", "sum"),
        )
        .reset_index()
    )

    if is_train:
        # only merge with train_df if it’s training data
        provider_summary = pd.merge(
            provider_summary, train_df, on="Provider", how="inner"
        )
        provider_summary["PotentialFraud"] = provider_summary["PotentialFraud"].map(
            {"Yes": 1, "No": 0}
        )

    return provider_summary


@st.cache_resource
def cache_model(_model):
    """Cache trained model in memory."""
    return _model


# -------------------------------------------------------------
# 📂 Sidebar: Data Upload Section
# -------------------------------------------------------------
st.sidebar.header("📂 Upload Datasets")
inpatient_file = st.sidebar.file_uploader("Inpatient CSV", type=["csv"])
outpatient_file = st.sidebar.file_uploader("Outpatient CSV", type=["csv"])
beneficiary_file = st.sidebar.file_uploader("Beneficiary CSV", type=["csv"])
train_file = st.sidebar.file_uploader("Train CSV", type=["csv"])

# -------------------------------------------------------------
# Workflow Tabs
# -------------------------------------------------------------
if inpatient_file and outpatient_file and beneficiary_file and train_file:
    inpatient_df, outpatient_df, beneficiary_df, train_df = load_data(
        inpatient_file, outpatient_file, beneficiary_file, train_file
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 EDA",
            "🧩 Feature Engineering",
            "🧮 Model Training",
            "📈 Evaluation",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1 — EDA
    # -------------------------------------------------------------
    with tab1:
        st.subheader("📊 Exploratory Data Analysis")
        st.write("### Dataset Overview")
        st.write(
            f"Inpatient: {inpatient_df.shape}, Outpatient: {outpatient_df.shape}, Beneficiary: {beneficiary_df.shape}"
        )

        st.write("### Sample Inpatient Data")
        st.dataframe(inpatient_df.head())

        if "DOB" in beneficiary_df.columns:
            beneficiary_df["Age"] = (
                dt.datetime.now().year - pd.to_datetime(beneficiary_df["DOB"]).dt.year
            )
            fig, ax = plt.subplots()
            sns.histplot(beneficiary_df["Age"], kde=True, bins=30, ax=ax)
            plt.title("Beneficiary Age Distribution")
            st.pyplot(fig)

        if "PotentialFraud" in train_df.columns:
            st.write("### Fraud vs Non-Fraud Providers")
            fig, ax = plt.subplots()
            train_df["PotentialFraud"].value_counts().plot.bar(
                color=["skyblue", "salmon"], ax=ax
            )
            plt.title("Fraud Distribution")
            st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 2 — Feature Engineering
    # -------------------------------------------------------------
    with tab2:
        st.subheader("🧩 Feature Engineering and Provider Summary")
        provider_summary = preprocess_and_engineer(
            inpatient_df,
            outpatient_df,
            beneficiary_df,
            train_df,
            is_train=True,
        )
        st.dataframe(provider_summary.head())
        st.success("✅ Features engineered successfully!")

        st.markdown(
            """
### 🧩 **Feature Creation Overview**

This section summarizes how raw claim and beneficiary data are transformed into meaningful features used for fraud detection.

#### **Feature Categories**

- **Procedures:** Columns representing medical procedure codes performed on patients.  
- **Diagnosis:** Columns capturing diagnosis codes assigned to patients (excluding admitting diagnosis).  
- **Chronic Conditions:** Binary indicators showing whether a beneficiary has chronic diseases.  
- **all_columns:** A combined list of all procedure, diagnosis, and chronic condition columns for simplified transformations.

#### **Derived Features**

| **Feature** | **Description** | **Analytical Purpose** |
|--------------|-----------------|------------------------|
| `procedureCount` | Total number of procedures associated with a claim | High counts may indicate **over-treatment** or **upcoding** |
| `diagnosisCount` | Total number of diagnoses linked to a claim | Excessive diagnoses may signal **claim padding** |
| `chronicCondCount` | Number of chronic conditions (after converting `2 → 0` for binary encoding) | Helps differentiate **justified claims** (many conditions) vs **potentially inflated ones** |

#### **Fraud-Indicative Flags**

- **Inpatient Anomaly:** Flags cases where `Part A Coverage = 0` but **inpatient reimbursements** exist — possible inpatient billing without valid coverage.  
- **Outpatient Anomaly:** Flags cases where `Part B Coverage = 0` but **outpatient reimbursements** exist — suggests fraudulent outpatient claims.  
- **No Physician Flag:** Identifies claims **without an attending physician** yet having reimbursements — potential fabricated or incomplete claims.

These engineered features form the foundation for detecting unusual provider behavior patterns in subsequent modeling steps.
"""
        )

        # -------------------------------------------------------------
        # TAB 3 — Model Training (with Train Evaluation)
        # -------------------------------------------------------------
    with tab3:
        st.subheader("🧮 Model Training — Random Forest + SMOTE + GridSearchCV")

        X = provider_summary.drop(columns=["PotentialFraud"])
        Y = provider_summary["PotentialFraud"]
        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, stratify=Y, random_state=43
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=43)),
                ("rf", RandomForestClassifier(random_state=43)),
            ]
        )

        param_grid = {
            "rf__n_estimators": [400],
            "rf__max_depth": [5],
            "rf__min_samples_split": [10],
            "rf__min_samples_leaf": [2],
            "rf__max_features": ["sqrt"],
        }

        with st.spinner("⏳ Running GridSearchCV..."):
            grid_search = GridSearchCV(
                pipeline, param_grid, scoring="roc_auc", cv=3, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

        best_model = cache_model(grid_search.best_estimator_)
        st.success(f"✅ Best Parameters Found: {grid_search.best_params_}")

        # ---------------------------------------------------------
        # Train Evaluation Section
        # ---------------------------------------------------------
        st.write("### 📈 Evaluating Model on Training Data")

        y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
        precision, recall, thresholds = precision_recall_curve(
            y_train, y_train_pred_prob
        )
        f1 = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1)]
        y_train_pred_class = (y_train_pred_prob >= best_threshold).astype(int)

        auc = roc_auc_score(y_train, y_train_pred_prob)
        acc = accuracy_score(y_train, y_train_pred_class)
        prec = precision_score(y_train, y_train_pred_class)
        rec = recall_score(y_train, y_train_pred_class)
        f1s = f1_score(y_train, y_train_pred_class)

        # Display Metrics in Streamlit
        st.write(f"**Accuracy:**  {round(acc*100,1)}%")
        st.write(f"**Precision:** {round(prec*100,1)}%")
        st.write(f"**Recall:**    {round(rec*100,1)}%")
        st.write(f"**F1-Score:**  {round(f1s*100)}%")
        st.write(f"**AUC:**       {round(auc*100,1)}%")
        st.write(f"**Optimal Cutoff Probability:** {best_threshold:.4f}")

        # Confusion Matrix
        st.write("#### Confusion Matrix — Train Data")
        cm = confusion_matrix(y_train, y_train_pred_class)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
            ax=ax,
        )
        plt.title("Confusion Matrix - Random Forest (Train Data)")
        st.pyplot(fig)

        # ROC Curve
        st.write("#### ROC Curve — Train Data")
        fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Random Forest with SMOTE (Train Data)")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        rf_model = best_model.named_steps["rf"]
        feat_imp = (
            pd.Series(rf_model.feature_importances_, index=X_train.columns)
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots()
        sns.barplot(
            x=feat_imp.values,
            y=feat_imp.index,
            hue=feat_imp.index,
            legend=False,
            palette="viridis",
            ax=ax,
        )
        plt.title("Top 10 Feature Importances")
        st.pyplot(fig)

        st.info(
            "✅ Model trained and evaluated successfully on training data. Proceed to Evaluation tab for test results."
        )

    # -------------------------------------------------------------
    # TAB 4 — Evaluation
    # -------------------------------------------------------------
    with tab4:
        st.subheader("📈 Model Evaluation")

        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        f1 = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1)]
        y_pred_class = (y_pred_prob >= best_threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_class)
        prec = precision_score(y_test, y_pred_class)
        rec = recall_score(y_test, y_pred_class)
        f1s = f1_score(y_test, y_pred_class)
        auc = roc_auc_score(y_test, y_pred_prob)

        st.write(f"**Accuracy:**  {round(acc*100,1)}%")
        st.write(f"**Precision:** {round(prec*100,1)}%")
        st.write(f"**Recall:**    {round(rec*100,1)}%")
        st.write(f"**F1-Score:**  {round(f1s*100)}%")
        st.write(f"**AUC:**       {round(auc*100,1)}%")
        st.write(f"**Optimal Cutoff:** {best_threshold:.4f}")

        cm = confusion_matrix(y_test, y_pred_class)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
            ax=ax,
        )
        st.pyplot(fig)

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("👈 Please upload all required CSVs to begin.")
