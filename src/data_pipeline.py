import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel("D:\\CustomerChurn\\data\\raw\\Bank_Churn_Messy.xlsx")
print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df1 = pd.read_excel("D:\\CustomerChurn\\data\\raw\\Bank_Churn_Messy.xlsx",sheet_name="Account_Info")
print(df1.head())
print(df1.info())
print(df1.shape)
print(df1.describe())
print(df1.isnull().sum())
print(df1.duplicated().sum())
##Categorical Columns Analysis
print(df['Geography'].value_counts())
print(df['Gender'].value_counts())
print(df1['IsActiveMember'].value_counts())
print(df1['HasCrCard'].value_counts())
print(df1['Exited'].value_counts())
## Numerical Columns Analysis
print(df['CreditScore'].describe())
print(df['Age'].describe())
print(df['Tenure'].describe())
print(df1['NumOfProducts'].describe())
print(df1['Exited'].describe())

##Visualization
plt.figure(figsize=(10,6))
sns.histplot(df['CreditScore'],bins=30,kde=True)
plt.title('Distribution of Credit Score')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Tenure',data=df)
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='NumOfProducts',data=df1)
plt.title("Product Distribution")
plt.xlabel("Number of Products")
plt.ylabel("Number of Customers")
plt.show()

plt.figure(figsize=(6, 5))
sns.countplot(data=df1, x="Exited")
plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

analysis_df = df.merge(df1,on='CustomerId',how='inner')
print(analysis_df.head())
print(analysis_df.shape)
print(analysis_df["Exited"].value_counts())
##Categoricals vs Churn
geo_churn = pd.crosstab(analysis_df['Geography'], analysis_df['Exited'], normalize='index')*100
print(geo_churn)

gender_churn = pd.crosstab(
    analysis_df["Gender"],
    analysis_df["Exited"],
    normalize="index"
) * 100

print(gender_churn)

status_churn = pd.crosstab(
    analysis_df["IsActiveMember"],
    analysis_df["Exited"],
    normalize="index"
) * 100

print(status_churn)

card_churn = pd.crosstab(
    analysis_df["HasCrCard"],
    analysis_df["Exited"],
    normalize="index"
) * 100

print(card_churn)

print(
    "IsActiveMember and HasCrCard identical:",
    (analysis_df["IsActiveMember"] == analysis_df["HasCrCard"]).all()
)

##Numerical VS Churn
print(analysis_df.columns.tolist())
print(
    "Tenure columns identical:",
    (analysis_df["Tenure_x"] == analysis_df["Tenure_y"]).all()
)

print(
    "Different Tenure values:",
    (analysis_df["Tenure_x"] != analysis_df["Tenure_y"]).sum()
)
print(
    analysis_df.groupby("Exited")[[
        "CreditScore",
        "Age",
        "Tenure_x",
        "NumOfProducts"
    ]].mean()
)
print(
    pd.crosstab(
        analysis_df["NumOfProducts"],
        analysis_df["Exited"],
        normalize="index"
    ) * 100
)
print(analysis_df['NumOfProducts'].value_counts().sort_index())
print(
    "IsActiveMember and HasCrCard identical:",
    (analysis_df["IsActiveMember"] == analysis_df["HasCrCard"]).all()
)
print(df[df['Surname'].isna()])
print(df[df['Age'].isna()])

print(df['Surname'].isna().sum())
print(df['Age'].isna().sum())
print(df['Age'].isna().mean()*100,2)

print(df[df.duplicated(keep=False)])
print(df1[df1.duplicated(keep=False)])

print(df['CustomerId'].duplicated().sum())
print(df1['CustomerId'].duplicated().sum())

df_clean = df.drop_duplicates().copy()
df1_clean = df1.drop_duplicates().copy()

print(len(df), '->',len(df_clean))
print(len(df1), '->',len(df1_clean))
print(type(df_clean))
print(type(df1_clean))
print(df_clean.shape)
print(df1_clean.shape)

print(
    "Suspicious EstimatedSalary values:",
    (df_clean["EstimatedSalary"] == "-€999999").sum()
)

print(
    "Suspicious Balance values:",
    (df1_clean["Balance"] == "-€999999").sum()
)

print(
    df_clean[df_clean["EstimatedSalary"] == "-€999999"]
)


df_clean["EstimatedSalary"] = df_clean["EstimatedSalary"].replace(
    "-€999999",
    np.nan
)
print(
    "Missing EstimatedSalary:",
    df_clean["EstimatedSalary"].isna().sum()
)
df_clean['EstimatedSalary'] = df_clean['EstimatedSalary'].str.replace('€',"",regex=False).astype(float)
print(df_clean['EstimatedSalary'].head())
print(df_clean['EstimatedSalary'].describe())
print(df_clean['EstimatedSalary'].median())
df_clean["EstimatedSalary"] = df_clean["EstimatedSalary"].fillna(
    df_clean["EstimatedSalary"].median()
)
print(df_clean["EstimatedSalary"].isna().sum())
print("Median Age:", df_clean["Age"].median())
df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)
print("Missing Age:", df_clean["Age"].isna().sum())
df_clean["Surname"] = df_clean["Surname"].fillna("Unknown")
print("Missing Surname:", df_clean["Surname"].isna().sum())

print(df1_clean["Balance"].dtype)
df1_clean["Balance"] = (
    df1_clean["Balance"]
    .str.replace("€", "", regex=False)
    .astype(float)
)
print(df1_clean["Balance"].head())
print("Missing Balance:", df1_clean["Balance"].isna().sum())

print("HasCrCard:", df1_clean["HasCrCard"].unique())
print("IsActiveMember:", df1_clean["IsActiveMember"].unique())
df1_clean["HasCrCard"] = df1_clean["HasCrCard"].map({
    "Yes": 1,
    "No": 0
})
print(df1_clean["HasCrCard"].value_counts())
df1_clean["IsActiveMember"] = df1_clean["IsActiveMember"].map({
    "Yes": 1,
    "No": 0
})
print(df1_clean["IsActiveMember"].unique())
print(df1_clean["IsActiveMember"].value_counts())

print(
    (df1_clean["HasCrCard"] == df1_clean["IsActiveMember"]).all()
)

print(df_clean["Geography"].value_counts())

df_clean["Geography"] = df_clean["Geography"].replace({
    "French": "France",
    "FRA": "France"
})


print(df_clean["Geography"].value_counts())
print(df_clean.dtypes)
print()
print(df1_clean.dtypes)

print("Customer_Info shape:", df_clean.shape)
print("Account_Info shape:", df1_clean.shape)

print("\nMissing values - Customer_Info:")
print(df_clean.isna().sum())

print("\nMissing values - Account_Info:")
print(df1_clean.isna().sum())

print("\nDuplicate rows - Customer_Info:", df_clean.duplicated().sum())
print("Duplicate rows - Account_Info:", df1_clean.duplicated().sum())

print("Customer_Info duplicate IDs:",
      df_clean["CustomerId"].duplicated().sum())

print("Account_Info duplicate IDs:",
      df1_clean["CustomerId"].duplicated().sum())
final_df = df_clean.merge(
    df1_clean,
    on="CustomerId",
    how="inner"
)
print(final_df.shape)
print(final_df.head())
print((final_df["Tenure_x"] == final_df["Tenure_y"]).all())

final_df = final_df.drop(columns=["Tenure_y"])

final_df = final_df.rename(columns={
    "Tenure_x": "Tenure"
})
print(final_df.shape)
print(final_df.columns.tolist())

print("Shape:", final_df.shape)

print("\nMissing values:")
print(final_df.isna().sum())

print("\nDuplicate rows:", final_df.duplicated().sum())

print("\nDuplicate CustomerIds:",
      final_df["CustomerId"].duplicated().sum())

print("\nExited values:")
print(final_df["Exited"].value_counts())

print("\nGeography:")
print(final_df["Geography"].value_counts())



# -------------------------------
# Churn Analysis
# -------------------------------

churn_count = final_df["Exited"].sum()
total_customers = len(final_df)

churn_rate = churn_count / total_customers * 100

print(f"Churn Count: {churn_count}")
print(f"Total Customers: {total_customers}")
print(f"Churn Rate: {churn_rate:.2f}%")

##Churn by age
final_df["AgeGroup"] = pd.cut(final_df['Age'],bins=[0,30,40,50,60,100],labels = ['18-30','31-40','41-50','51-60','61+'])
churn_by_age = final_df.groupby("AgeGroup",observed=True)['Exited'].mean() * 100
print("\nChurn Rate by Age Group:")
print(churn_by_age.round(2))

##churn rate by geography


geo_churn = final_df.groupby("Geography")["Exited"].mean() * 100

print("\nChurn Rate by Geography:")
print(geo_churn.round(2))

##churn rate by gender
gender_churn = final_df.groupby("Gender")['Exited'].mean() * 100
print("\nChurn Rate by Gender:")
print(gender_churn.round(2))

# Churn rate by Active Membership

active_churn = final_df.groupby("IsActiveMember")["Exited"].mean() * 100

print("\nChurn Rate by Active Membership:")
print(active_churn.round(2))

# Churn rate by Number of Products

product_churn = final_df.groupby("NumOfProducts")["Exited"].mean() * 100

print("\nChurn Rate by Number of Products:")
print(product_churn.round(2))
# Churn rate by Credit Score Group

final_df["CreditScoreGroup"] = pd.cut(
    final_df["CreditScore"],
    bins=[0, 500, 600, 700, 800, 900],
    labels=["<500", "500-600", "601-700", "701-800", "801+"]
)

credit_churn = final_df.groupby(
    "CreditScoreGroup",
    observed=True
)["Exited"].mean() * 100

print("\nChurn Rate by Credit Score Group:")
print(credit_churn.round(2))

# Customers aged 51-60 with Credit Score below 500

count = final_df[
    (final_df["Age"].between(51, 60)) &
    (final_df["CreditScore"] < 500)
].shape[0]

print("51-60 age group with Credit Score <500:", count)

group = final_df[
    (final_df["Age"].between(51, 60)) &
    (final_df["CreditScore"] < 500)
]

print("Total customers:", len(group))
print("Churned customers:", group["Exited"].sum())
print("Churn rate:", round(group["Exited"].mean() * 100, 2), "%")

group = final_df[
    (final_df["Age"].between(51, 60)) &
    (final_df["CreditScore"] >= 500)
]

print("Total customers:", len(group))
print("Churned customers:", group["Exited"].sum())
print("Churn rate:", round(group["Exited"].mean() * 100, 2), "%")

# Churn rate by Balance Group

final_df["BalanceGroup"] = pd.cut(
    final_df["Balance"],
    bins=[-1, 50000, 100000, 150000, 200000],
    labels=["0-50K", "50K-100K", "100K-150K", "150K+"]
)

balance_churn = final_df.groupby(
    "BalanceGroup",
    observed=True
)["Exited"].mean() * 100

print("\nChurn Rate by Balance Group:")
print(balance_churn.round(2))

balance_counts = final_df["BalanceGroup"].value_counts().sort_index()

print("\nCustomers by Balance Group:")
print(balance_counts)

age_balance_churn = final_df.groupby(
    ["AgeGroup", "BalanceGroup"],
    observed=True
)["Exited"].mean() * 100

print("\nChurn Rate by Age and Balance:")
print(age_balance_churn.round(2))

geo_age = pd.crosstab(
    final_df["Geography"],
    final_df["AgeGroup"],
    normalize="index"
) * 100

print("\nAge Distribution by Geography:")
print(geo_age.round(2))

geo_balance = pd.crosstab(
    final_df["Geography"],
    final_df["BalanceGroup"],
    normalize="index"
) * 100

print("\nBalance Distribution by Geography:")
print(geo_balance.round(2))

geo_age_churn = final_df.groupby(
    ["AgeGroup", "Geography"],
    observed=True
)["Exited"].mean() * 100

print("\nChurn Rate by Age Group and Geography:")
print(geo_age_churn.round(2))

geo_age_balance_churn = final_df.groupby(
    ["AgeGroup", "BalanceGroup", "Geography"],
    observed=True
)["Exited"].mean() * 100

print("\nChurn Rate by Age, Balance and Geography:")
print(geo_age_balance_churn.round(2))

geo_age_balance_count = final_df.groupby(
    ["AgeGroup", "BalanceGroup", "Geography"],
    observed=True
).size()

print("\nCustomer Count by Age, Balance and Geography:")
print(geo_age_balance_count)

geo_age_balance_count = final_df.groupby(
    ["AgeGroup", "BalanceGroup", "Geography"],
    observed=True
).size()

print("\nCustomer Count by Age, Balance and Geography:")
print(geo_age_balance_count)

geo_activity_churn = final_df.groupby(
    ["IsActiveMember", "Geography"]
)["Exited"].mean() * 100

print("\nChurn Rate by Activity and Geography:")
print(geo_activity_churn.round(2))

germany_churn = final_df[
    (final_df["Geography"] == "Germany") &
    (final_df["Exited"] == 1)
]

print("German churned customers:", len(germany_churn))

print("\nAverage characteristics:")
print(
    germany_churn[
        ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
         "HasCrCard", "IsActiveMember"]
    ].mean().round(2)
)

germany_stayed = final_df[
    (final_df["Geography"] == "Germany") &
    (final_df["Exited"] == 0)
]

print("German stayed customers:", len(germany_stayed))

print("\nAverage characteristics:")
print(
    germany_stayed[
        ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
         "HasCrCard", "IsActiveMember"]
    ].mean().round(2)
)

germany_age_activity = final_df[
    final_df["Geography"] == "Germany"
]

age_activity_churn = germany_age_activity.groupby(
    ["AgeGroup", "IsActiveMember"],
    observed=True
)["Exited"].mean() * 100

print("\nGermany - Churn Rate by Age Group and Activity:")
print(age_activity_churn.round(2))

germany_age_activity_count = germany_age_activity.groupby(
    ["AgeGroup", "IsActiveMember"],
    observed=True
).agg(
    Customers=("CustomerId", "count"),
    Churned=("Exited", "sum"),
    ChurnRate=("Exited", "mean")
)

germany_age_activity_count["ChurnRate"] *= 100

print(germany_age_activity_count.round(2))

segment_analysis = final_df.groupby(
    ["Geography", "AgeGroup", "IsActiveMember"],
    observed=True
).agg(
    Customers=("CustomerId", "count"),
    Churned=("Exited", "sum"),
    ChurnRate=("Exited", "mean")
)

segment_analysis["ChurnRate"] *= 100

segment_analysis = segment_analysis.sort_values(
    "Churned",
    ascending=False
)

print("\nTop segments by number of churned customers:")
print(segment_analysis.head(10).round(2))

final_df.to_csv(
    "data/processed/customer_churn_cleaned.csv",
    index=False
)