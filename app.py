# fraud_dashboard.py

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 EDA",
            "🧩 Feature Engineering",
            "🧮 Model Training",
            "📈 Evaluation",
            "🔮 Unseen Predictions",
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

        # rf_model = best_model.named_steps["rf"]
        # feat_imp = (
        #     pd.Series(rf_model.feature_importances_, index=X_train.columns)
        #     .sort_values(ascending=False)
        #     .head(10)
        # )
        # fig, ax = plt.subplots()
        # sns.barplot(
        #     x=feat_imp.values,
        #     y=feat_imp.index,
        #     hue=feat_imp.index,
        #     legend=False,
        #     palette="viridis",
        #     ax=ax,
        # )
        # plt.title("Top 10 Feature Importances")
        # st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 5 — Unseen Predictions
    # -------------------------------------------------------------
    with tab5:
        st.subheader("🔮 Predict on Unseen Data")
        unseen_inp = st.file_uploader("Upload Unseen Inpatient CSV", type=["csv"])
        unseen_out = st.file_uploader("Upload Unseen Outpatient CSV", type=["csv"])
        unseen_bene = st.file_uploader("Upload Unseen Beneficiary CSV", type=["csv"])
        unseen_main = st.file_uploader("Upload Unseen Provider CSV", type=["csv"])

        if unseen_inp and unseen_out and unseen_bene and unseen_main:
            st.success("✅ Unseen datasets uploaded successfully!")

            unseen_inp_df = pd.read_csv(unseen_inp)
            unseen_out_df = pd.read_csv(unseen_out)
            unseen_bene_df = pd.read_csv(unseen_bene)
            unseen_main_df = pd.read_csv(unseen_main)

            unseen_provider_summary = preprocess_and_engineer(
                unseen_inp_df,
                unseen_out_df,
                unseen_bene_df,
                unseen_main_df,
                is_train=False,
            )

            unseen_provider_summary = unseen_provider_summary.select_dtypes(
                include=[np.number]
            )

            y_unseen_pred_prob = best_model.predict_proba(unseen_provider_summary)[:, 1]
            y_unseen_pred_class = (y_unseen_pred_prob >= best_threshold).astype(int)

            unseen_provider_summary["FraudProbability"] = y_unseen_pred_prob
            unseen_provider_summary["PredictedFraud"] = y_unseen_pred_class

            st.dataframe(unseen_provider_summary.head(10))

            fig, ax = plt.subplots()
            sns.histplot(
                unseen_provider_summary["FraudProbability"],
                bins=20,
                kde=True,
                color="royalblue",
                ax=ax,
            )
            plt.title("Distribution of Predicted Fraud Probabilities")
            st.pyplot(fig)

            csv_download = unseen_provider_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                csv_download,
                "fraud_predictions.csv",
                "text/csv",
            )

else:
    st.info("👈 Please upload all required CSVs to begin.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime as dt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     confusion_matrix,
#     roc_auc_score,
#     roc_curve,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     precision_recall_curve,
# )
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline

# # -------------------------------------------------------------
# # 🎨 Streamlit Page Setup
# # -------------------------------------------------------------
# st.set_page_config(page_title="Healthcare Provider Fraud Detection", layout="wide")
# st.title("🏥 Healthcare Provider Fraud Detection Dashboard")
# st.markdown(
#     "A complete pipeline for analyzing healthcare claims, detecting fraudulent providers, and visualizing key data insights."
# )

# st.sidebar.header("📂 Upload Required Datasets")

# # -------------------------------------------------------------
# # 📂 File Upload Section
# # -------------------------------------------------------------
# inpatient_file = st.sidebar.file_uploader("Upload Inpatient Data CSV", type=["csv"])
# outpatient_file = st.sidebar.file_uploader("Upload Outpatient Data CSV", type=["csv"])
# beneficiary_file = st.sidebar.file_uploader("Upload Beneficiary Data CSV", type=["csv"])
# train_file = st.sidebar.file_uploader("Upload Training Data CSV", type=["csv"])
# unseen_option = st.sidebar.checkbox("Do you also want to predict on unseen data?")

# # -------------------------------------------------------------
# # 🚀 Process Files
# # -------------------------------------------------------------
# if inpatient_file and outpatient_file and beneficiary_file and train_file:
#     inpatient_df = pd.read_csv(inpatient_file)
#     outpatient_df = pd.read_csv(outpatient_file)
#     beneficiary_df = pd.read_csv(beneficiary_file)
#     train_df = pd.read_csv(train_file)

#     st.success("✅ All datasets uploaded successfully!")

#     # Tabs for workflow sections
#     tab1, tab2, tab4, tab5 = st.tabs(
#         [
#             "📊 EDA Summary",
#             "🧩 Feature Engineering",
#             # "🧮 Model Training",
#             "📈 Model Evaluation",
#             "🔮 Unseen Predictions",
#         ]
#     )

#     # -------------------------------------------------------------
#     # 🧾 TAB 1 — EDA Summary
#     # -------------------------------------------------------------
#     with tab1:
#         st.subheader("📊 Data Overview")
#         st.write("### Dataset Shapes")
#         st.write(
#             f"Inpatient: {inpatient_df.shape}, Outpatient: {outpatient_df.shape}, Beneficiary: {beneficiary_df.shape}, Train: {train_df.shape}"
#         )

#         st.write("### Missing Value Summary")
#         missing_summary = pd.DataFrame(
#             {
#                 "Inpatient": inpatient_df.isnull().sum(),
#                 "Outpatient": outpatient_df.isnull().sum(),
#                 "Beneficiary": beneficiary_df.isnull().sum(),
#             }
#         )
#         st.dataframe(missing_summary)

#         st.write("### Sample Inpatient Data")
#         st.dataframe(inpatient_df.head())

#         # Age Distribution
#         st.write("### Beneficiary Age Distribution")
#         if "DOB" in beneficiary_df.columns:
#             beneficiary_df["Age"] = (
#                 dt.datetime.now().year - pd.to_datetime(beneficiary_df["DOB"]).dt.year
#             )
#             fig, ax = plt.subplots()
#             sns.histplot(beneficiary_df["Age"], kde=True, bins=30, ax=ax)
#             plt.title("Beneficiary Age Distribution")
#             st.pyplot(fig)

#         # Race Distribution
#         if "Race" in beneficiary_df.columns:
#             st.write("### Beneficiary Race Distribution")
#             fig, ax = plt.subplots()
#             beneficiary_df["Race"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
#             plt.title("Race Distribution")
#             st.pyplot(fig)

#         # Reimbursement Distribution
#         st.write("### Claim Amount Distribution (Inpatient vs Outpatient)")
#         fig, ax = plt.subplots()
#         sns.boxplot(
#             data=pd.concat(
#                 [
#                     inpatient_df.assign(Type="Inpatient")[
#                         ["InscClaimAmtReimbursed", "Type"]
#                     ],
#                     outpatient_df.assign(Type="Outpatient")[
#                         ["InscClaimAmtReimbursed", "Type"]
#                     ],
#                 ]
#             ),
#             x="Type",
#             y="InscClaimAmtReimbursed",
#             ax=ax,
#         )
#         plt.title("Distribution of Reimbursed Amounts")
#         st.pyplot(fig)

#         # Fraud Count
#         if "PotentialFraud" in train_df.columns:
#             st.write("### Fraud vs Non-Fraud Providers")
#             fig, ax = plt.subplots()
#             train_df["PotentialFraud"].value_counts().plot.bar(
#                 ax=ax, color=["skyblue", "salmon"]
#             )
#             plt.title("Fraud Distribution")
#             plt.xlabel("Fraud Label")
#             plt.ylabel("Provider Count")
#             st.pyplot(fig)

#         st.success("✅ EDA Summary Completed!")

#     # -------------------------------------------------------------
#     # 🧩 TAB 2 — Feature Engineering
#     # -------------------------------------------------------------
#     with tab2:
#         st.subheader("🧩 Feature Engineering and Provider Summary")

#         # Convert date columns
#         date_columns = [col for col in inpatient_df.columns if "dt" in col.lower()]
#         for date_col in date_columns:
#             if date_col in inpatient_df.columns:
#                 inpatient_df[date_col] = pd.to_datetime(inpatient_df[date_col])
#             if date_col in outpatient_df.columns:
#                 outpatient_df[date_col] = pd.to_datetime(outpatient_df[date_col])

#         for data in [inpatient_df, outpatient_df, beneficiary_df]:
#             data.drop_duplicates(inplace=True)

#         claims_df = pd.concat([inpatient_df, outpatient_df], axis=0)
#         claims_df = pd.merge(claims_df, beneficiary_df, on=["BeneID"], how="left")

#         claims_df["daysAdmitted"] = (
#             claims_df["DischargeDt"] - claims_df["AdmissionDt"]
#         ).dt.days.fillna(0)
#         claims_df["beneAge"] = np.where(
#             claims_df["DOD"].isnull(),
#             dt.datetime.now().year - pd.to_datetime(claims_df["DOB"]).dt.year,
#             pd.to_datetime(claims_df["DOD"]).dt.year
#             - pd.to_datetime(claims_df["DOB"]).dt.year,
#         )

#         # Procedure/Diagnosis Features
#         procedures = [c for c in claims_df.columns if "procedurecode" in c.lower()]
#         diagnosis = [
#             c
#             for c in claims_df.columns
#             if "diagnosiscode" in c.lower() and "clmadmitdiagnosiscode" not in c.lower()
#         ]
#         chronic_cond = [c for c in claims_df.columns if "chroniccond" in c.lower()]
#         all_columns = procedures + diagnosis + chronic_cond

#         claims_df["procedureCount"] = claims_df[procedures].count(axis=1)
#         claims_df["diagnosisCount"] = claims_df[diagnosis].count(axis=1)
#         claims_df[chronic_cond] = claims_df[chronic_cond].replace({2: 0})
#         claims_df["chronicCondCount"] = claims_df[chronic_cond].sum(axis=1)

#         # Anomaly Features
#         claims_df["inpatientAnomaly"] = np.where(
#             (claims_df["NoOfMonths_PartACov"] == 0)
#             & (claims_df["IPAnnualReimbursementAmt"] > 0),
#             1,
#             0,
#         )
#         claims_df["outpatientAnomaly"] = np.where(
#             (claims_df["NoOfMonths_PartBCov"] == 0)
#             & (claims_df["OPAnnualReimbursementAmt"] > 0),
#             1,
#             0,
#         )
#         claims_df["noPhysician"] = np.where(
#             (claims_df["AttendingPhysician"].isnull())
#             & (claims_df["InscClaimAmtReimbursed"] > 0),
#             1,
#             0,
#         )
#         claims_df["claimDuration"] = (
#             claims_df["ClaimEndDt"] - claims_df["ClaimStartDt"]
#         ).dt.days

#         cleaned_df = claims_df.drop(all_columns, axis=1, inplace=False)

#         provider_summary = (
#             cleaned_df.groupby(["Provider"])
#             .agg(
#                 numClaims=("ClaimID", "count"),
#                 totalClaimReimbersed=("InscClaimAmtReimbursed", "sum"),
#                 maxClaimReimbursed=("InscClaimAmtReimbursed", "max"),
#                 meanClaimReimbersed=("InscClaimAmtReimbursed", "mean"),
#                 ipAnnualReimbursementAmt=("IPAnnualReimbursementAmt", "mean"),
#                 opAnnualReimbursementAmt=("OPAnnualReimbursementAmt", "mean"),
#                 totalProcedures=("procedureCount", "sum"),
#                 totalDiagnosis=("diagnosisCount", "sum"),
#                 totalchronicCond=("chronicCondCount", "sum"),
#                 inpatientAnomaly=("inpatientAnomaly", "sum"),
#                 outpatientAnomaly=("outpatientAnomaly", "sum"),
#                 noPhysician=("noPhysician", "sum"),
#             )
#             .reset_index()
#         )

#         provider_summary = pd.merge(
#             provider_summary, train_df, on="Provider", how="inner"
#         )
#         provider_summary["PotentialFraud"] = provider_summary["PotentialFraud"].map(
#             {"Yes": 1, "No": 0}
#         )
#         st.dataframe(provider_summary.head())

#         st.success("✅ Feature engineering and provider-level aggregation complete.")

#         # -------------------------------------------------------------
#         # 🧮 TAB 3 — Model Training
#         # -------------------------------------------------------------
#         # -------------------------------------------------------------
#         # 🧮 TAB 3 — Model Training (Fixed Version)
#         # -------------------------------------------------------------
#         # with tab3:
#         st.subheader("🧮 Model Training - Random Forest + SMOTE + GridSearchCV")

#         # Separate features and target
#         X = provider_summary.drop(columns=["PotentialFraud"])
#         Y = provider_summary["PotentialFraud"]

#         # ⚙️ FIX: Remove non-numeric columns before scaling
#         non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
#         if len(non_numeric_cols) > 0:
#             st.warning(f"⚠️ Dropping non-numeric columns: {', '.join(non_numeric_cols)}")
#             X = X.select_dtypes(include=[np.number])

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, Y, test_size=0.3, stratify=Y, random_state=43
#         )

#         # Define pipeline
#         pipeline = Pipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 ("smote", SMOTE(random_state=43)),
#                 ("rf", RandomForestClassifier(random_state=43)),
#             ]
#         )

#         # Hyperparameter grid
#         param_grid = {
#             "rf__n_estimators": [400],
#             "rf__max_depth": [5],
#             "rf__min_samples_split": [10],
#             "rf__min_samples_leaf": [2],
#             "rf__max_features": ["sqrt"],
#         }

#         # GridSearchCV with proper error handling
#         with st.spinner("⏳ Running GridSearchCV..."):
#             grid_search = GridSearchCV(
#                 estimator=pipeline,
#                 param_grid=param_grid,
#                 scoring="roc_auc",
#                 cv=3,
#                 n_jobs=-1,
#                 error_score="raise",  # helps debug if still any fit fails
#             )
#             grid_search.fit(X_train, y_train)

#         best_model = grid_search.best_estimator_
#         st.success(f"✅ Best Parameters Found: {grid_search.best_params_}")

#         # -------------------------------------------------------------
#         # Evaluate model on test data
#         # -------------------------------------------------------------
#         y_pred_prob = best_model.predict_proba(X_test)[:, 1]

#         precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
#         f1 = 2 * (precision * recall) / (precision + recall)
#         best_threshold = thresholds[np.argmax(f1)]

#         y_pred_class = (y_pred_prob >= best_threshold).astype(int)

#         acc = accuracy_score(y_test, y_pred_class)
#         prec = precision_score(y_test, y_pred_class)
#         rec = recall_score(y_test, y_pred_class)
#         f1s = f1_score(y_test, y_pred_class)
#         auc = roc_auc_score(y_test, y_pred_prob)

#         st.write(f"**Accuracy:** {acc:.4f}")
#         st.write(f"**Precision:** {prec:.4f}")
#         st.write(f"**Recall:** {rec:.4f}")
#         st.write(f"**F1 Score:** {f1s:.4f}")
#         st.write(f"**ROC-AUC:** {auc:.4f}")
#         st.write(f"**Optimal Cutoff Probability:** {best_threshold:.4f}")

#         # Confusion Matrix
#         cm = confusion_matrix(y_test, y_pred_class)
#         fig, ax = plt.subplots()
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt="d",
#             cmap="Blues",
#             xticklabels=["Not Fraud", "Fraud"],
#             yticklabels=["Not Fraud", "Fraud"],
#             ax=ax,
#         )
#         ax.set_title("Confusion Matrix - Test Data")
#         st.pyplot(fig)

#         # ROC Curve
#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#         fig, ax = plt.subplots()
#         ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
#         ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
#         ax.set_xlabel("False Positive Rate")
#         ax.set_ylabel("True Positive Rate")
#         ax.legend()
#         st.pyplot(fig)

#         # Feature Importance
#         st.subheader("🏆 Feature Importance")
#         rf_model = best_model.named_steps["rf"]
#         feat_imp = (
#             pd.Series(rf_model.feature_importances_, index=X_train.columns)
#             .sort_values(ascending=False)
#             .head(10)
#         )
#         fig, ax = plt.subplots()
#         sns.barplot(
#             x=feat_imp.values,
#             y=feat_imp.index,
#             hue=feat_imp.index,
#             legend=False,
#             palette="viridis",
#             ax=ax,
#         )
#         st.pyplot(fig)

#     # with tab3:
#     #     st.subheader("🧮 Model Training - Random Forest + SMOTE + GridSearch")

#     #     X = provider_summary.drop(columns=["PotentialFraud"])
#     #     Y = provider_summary["PotentialFraud"]

#     #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=43)

#     #     pipeline = Pipeline([
#     #         ("scaler", StandardScaler()),
#     #         ("smote", SMOTE(random_state=43)),
#     #         ("rf", RandomForestClassifier(random_state=43))
#     #     ])

#     #     param_grid = {
#     #         "rf__n_estimators": [400],
#     #         "rf__max_depth": [5],
#     #         "rf__min_samples_split": [10],
#     #         "rf__min_samples_leaf": [2],
#     #         'rf__max_features': 'sqrt',
#     #     }

#     #     with st.spinner("⏳ Running GridSearch..."):
#     #         grid_search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
#     #         grid_search.fit(X_train, y_train)

#     #     best_model = grid_search.best_estimator_
#     #     st.success(f"✅ Best Parameters: {grid_search.best_params_}")

#     # -------------------------------------------------------------
#     # 📈 TAB 4 — Model Evaluation
#     # -------------------------------------------------------------
#     with tab4:
#         st.subheader("📈 Model Evaluation Metrics")

#         y_pred_prob = best_model.predict_proba(X_test)[:, 1]
#         precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
#         f1 = 2 * (precision * recall) / (precision + recall)
#         best_threshold = thresholds[np.argmax(f1)]

#         y_pred_class = (y_pred_prob >= best_threshold).astype(int)

#         acc = accuracy_score(y_test, y_pred_class)
#         prec = precision_score(y_test, y_pred_class)
#         rec = recall_score(y_test, y_pred_class)
#         f1s = f1_score(y_test, y_pred_class)
#         auc = roc_auc_score(y_test, y_pred_prob)

#         st.write(f"**Accuracy:** {acc:.4f}")
#         st.write(f"**Precision:** {prec:.4f}")
#         st.write(f"**Recall:** {rec:.4f}")
#         st.write(f"**F1 Score:** {f1s:.4f}")
#         st.write(f"**ROC-AUC:** {auc:.4f}")
#         st.write(f"**Optimal Cutoff Probability:** {best_threshold:.4f}")

#         cm = confusion_matrix(y_test, y_pred_class)
#         fig, ax = plt.subplots()
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt="d",
#             cmap="Blues",
#             xticklabels=["Not Fraud", "Fraud"],
#             yticklabels=["Not Fraud", "Fraud"],
#             ax=ax,
#         )
#         ax.set_title("Confusion Matrix - Test Data")
#         st.pyplot(fig)

#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#         fig, ax = plt.subplots()
#         ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
#         ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
#         ax.set_xlabel("False Positive Rate")
#         ax.set_ylabel("True Positive Rate")
#         ax.legend()
#         st.pyplot(fig)

#         # Feature Importance
#         st.subheader("🏆 Feature Importance")
#         rf_model = best_model.named_steps["rf"]
#         feat_imp = (
#             pd.Series(rf_model.feature_importances_, index=X_train.columns)
#             .sort_values(ascending=False)
#             .head(10)
#         )
#         fig, ax = plt.subplots()
#         sns.barplot(
#             x=feat_imp.values,
#             y=feat_imp.index,
#             hue=feat_imp.index,
#             legend=False,
#             palette="viridis",
#             ax=ax,
#         )
#         st.pyplot(fig)

#     # -------------------------------------------------------------
#     # 🔮 TAB 5 — Unseen Predictions
#     # -------------------------------------------------------------
#     # -------------------------------------------------------------
#     # 🔮 TAB 5 — Unseen Predictions (Fixed + Functional)
#     # -------------------------------------------------------------
#     with tab5:
#         st.subheader("🔮 Predict on Unseen Data")
#         st.info(
#             "Upload unseen inpatient, outpatient, beneficiary, and provider CSVs to generate fraud probabilities using the trained model."
#         )

#         unseen_inp = st.file_uploader("Upload Unseen Inpatient CSV", type=["csv"])
#         unseen_out = st.file_uploader("Upload Unseen Outpatient CSV", type=["csv"])
#         unseen_bene = st.file_uploader("Upload Unseen Beneficiary CSV", type=["csv"])
#         unseen_main = st.file_uploader("Upload Unseen Provider CSV", type=["csv"])

#         if unseen_inp and unseen_out and unseen_bene and unseen_main:
#             st.success("✅ All unseen datasets uploaded successfully!")

#             unseen_inp_df = pd.read_csv(unseen_inp)
#             unseen_out_df = pd.read_csv(unseen_out)
#             unseen_bene_df = pd.read_csv(unseen_bene)
#             unseen_main_df = pd.read_csv(unseen_main)

#             # -------------------------------------------------------------
#             # 🧹 Preprocessing Unseen Data
#             # -------------------------------------------------------------
#             date_columns = [col for col in unseen_inp_df.columns if "dt" in col.lower()]
#             for date_col in date_columns:
#                 if date_col in unseen_inp_df.columns:
#                     unseen_inp_df[date_col] = pd.to_datetime(unseen_inp_df[date_col])
#                 if date_col in unseen_out_df.columns:
#                     unseen_out_df[date_col] = pd.to_datetime(unseen_out_df[date_col])

#             for data in [unseen_inp_df, unseen_out_df, unseen_bene_df]:
#                 data.drop_duplicates(inplace=True)

#             unseen_claims_df = pd.concat([unseen_inp_df, unseen_out_df], axis=0)
#             unseen_claims_df = pd.merge(
#                 unseen_claims_df, unseen_bene_df, on=["BeneID"], how="left"
#             )

#             unseen_claims_df["daysAdmitted"] = (
#                 unseen_claims_df["DischargeDt"] - unseen_claims_df["AdmissionDt"]
#             ).dt.days.fillna(0)
#             unseen_claims_df["beneAge"] = np.where(
#                 unseen_claims_df["DOD"].isnull(),
#                 dt.datetime.now().year
#                 - pd.to_datetime(unseen_claims_df["DOB"]).dt.year,
#                 pd.to_datetime(unseen_claims_df["DOD"]).dt.year
#                 - pd.to_datetime(unseen_claims_df["DOB"]).dt.year,
#             )

#             # -------------------------------------------------------------
#             # 🧩 Feature Engineering for Unseen Data
#             # -------------------------------------------------------------
#             procedures = [
#                 col
#                 for col in unseen_claims_df.columns
#                 if "procedurecode" in col.lower()
#             ]
#             diagnosis = [
#                 col
#                 for col in unseen_claims_df.columns
#                 if "diagnosiscode" in col.lower()
#             ]
#             chronic_cond = [
#                 col for col in unseen_claims_df.columns if "chroniccond" in col.lower()
#             ]
#             all_columns = procedures + diagnosis + chronic_cond

#             unseen_claims_df["procedureCount"] = unseen_claims_df[procedures].count(
#                 axis=1
#             )
#             unseen_claims_df["diagnosisCount"] = unseen_claims_df[diagnosis].count(
#                 axis=1
#             )
#             unseen_claims_df[chronic_cond] = unseen_claims_df[chronic_cond].replace(
#                 {2: 0}
#             )
#             unseen_claims_df["chronicCondCount"] = unseen_claims_df[chronic_cond].sum(
#                 axis=1
#             )

#             unseen_claims_df["inpatientAnomaly"] = np.where(
#                 (unseen_claims_df["NoOfMonths_PartACov"] == 0)
#                 & (unseen_claims_df["IPAnnualReimbursementAmt"] > 0),
#                 1,
#                 0,
#             )
#             unseen_claims_df["outpatientAnomaly"] = np.where(
#                 (unseen_claims_df["NoOfMonths_PartBCov"] == 0)
#                 & (unseen_claims_df["OPAnnualReimbursementAmt"] > 0),
#                 1,
#                 0,
#             )
#             unseen_claims_df["noPhysician"] = np.where(
#                 (unseen_claims_df["AttendingPhysician"].isnull())
#                 & (unseen_claims_df["InscClaimAmtReimbursed"] > 0),
#                 1,
#                 0,
#             )
#             unseen_claims_df["claimDuration"] = (
#                 unseen_claims_df["ClaimEndDt"] - unseen_claims_df["ClaimStartDt"]
#             ).dt.days

#             unseen_cleaned_df = unseen_claims_df.drop(
#                 all_columns, axis=1, inplace=False
#             )

#             unseen_provider_summary = (
#                 unseen_cleaned_df.groupby(["Provider"])
#                 .agg(
#                     numClaims=("ClaimID", "count"),
#                     totalClaimReimbersed=("InscClaimAmtReimbursed", "sum"),
#                     maxClaimReimbursed=("InscClaimAmtReimbursed", "max"),
#                     meanClaimReimbersed=("InscClaimAmtReimbursed", "mean"),
#                     ipAnnualReimbursementAmt=("IPAnnualReimbursementAmt", "mean"),
#                     opAnnualReimbursementAmt=("OPAnnualReimbursementAmt", "mean"),
#                     totalProcedures=("procedureCount", "sum"),
#                     totalDiagnosis=("diagnosisCount", "sum"),
#                     totalchronicCond=("chronicCondCount", "sum"),
#                     inpatientAnomaly=("inpatientAnomaly", "sum"),
#                     outpatientAnomaly=("outpatientAnomaly", "sum"),
#                     noPhysician=("noPhysician", "sum"),
#                 )
#                 .reset_index()
#             )

#             unseen_provider_summary.set_index("Provider", inplace=True)

#             # -------------------------------------------------------------
#             # ⚙️ FIX: Drop non-numeric columns
#             # -------------------------------------------------------------
#             non_numeric_cols = unseen_provider_summary.select_dtypes(
#                 exclude=[np.number]
#             ).columns.tolist()
#             if len(non_numeric_cols) > 0:
#                 st.warning(
#                     f"⚠️ Dropping non-numeric columns: {', '.join(non_numeric_cols)}"
#                 )
#                 unseen_provider_summary = unseen_provider_summary.select_dtypes(
#                     include=[np.number]
#                 )

#             # -------------------------------------------------------------
#             # 🔮 Predictions
#             # -------------------------------------------------------------
#             st.info("🚀 Generating Fraud Predictions on Unseen Data...")
#             y_unseen_pred_prob = best_model.predict_proba(unseen_provider_summary)[:, 1]
#             y_unseen_pred_class = (y_unseen_pred_prob >= best_threshold).astype(int)

#             unseen_provider_summary["FraudProbability"] = y_unseen_pred_prob
#             unseen_provider_summary["PredictedFraud"] = y_unseen_pred_class

#             # -------------------------------------------------------------
#             # 📊 Display & Download Results
#             # -------------------------------------------------------------
#             st.success("✅ Predictions generated successfully!")
#             st.write("### Top 10 Predicted Providers:")
#             st.dataframe(unseen_provider_summary.head(10))

#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.histplot(
#                 unseen_provider_summary["FraudProbability"],
#                 bins=20,
#                 kde=True,
#                 color="royalblue",
#                 ax=ax,
#             )
#             plt.title("Distribution of Predicted Fraud Probabilities")
#             st.pyplot(fig)

#             # Download predictions as CSV
#             csv_download = (
#                 unseen_provider_summary.reset_index()
#                 .to_csv(index=False)
#                 .encode("utf-8")
#             )
#             st.download_button(
#                 label="⬇️ Download Fraud Predictions as CSV",
#                 data=csv_download,
#                 file_name="unseen_fraud_predictions.csv",
#                 mime="text/csv",
#             )

# else:
#     st.info("👈 Please upload all required CSVs to begin.")
