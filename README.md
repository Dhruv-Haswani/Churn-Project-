# Customer Churn Analysis

An end-to-end customer churn analytics project using Python and Power BI.

The project starts with a messy banking customer dataset, performs data cleaning and exploratory analysis using Python, and then builds an interactive Power BI dashboard to understand customer churn patterns and identify high-risk customer segments.

---

## Project Overview

Customer churn is an important business problem for banks because understanding why customers leave can help identify groups that may require further attention.

This project focuses on:

- Understanding the raw customer data
- Cleaning and validating inconsistent data
- Performing exploratory data analysis
- Analyzing customer churn patterns
- Identifying high-risk customer segments
- Building an interactive Power BI dashboard

---

## Tech Stack

- **Python**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
- **Power BI**
- **Git & GitHub**
- **Excel / CSV**

---

## Dataset

The original dataset is provided as an Excel workbook containing two sheets:

- `Customer_Info`
- `Account_Info`

The two datasets are joined using `CustomerId`.

The target variable is:

- `Exited = 0` → Customer stayed
- `Exited = 1` → Customer churned

The final cleaned dataset contains **10,000 customers and 13 columns**.

---

## Data Cleaning

The raw data contained several quality issues that were handled using Python.

### Missing Values

- Missing `Surname` values were replaced with `Unknown`
- Missing `Age` values were filled using the median age
- Invalid salary placeholder values were converted to missing values and median-imputed

### Duplicate Records

Exact duplicate records were identified and removed from both source datasets.

### Data Type Cleaning

- Currency symbols were removed from `EstimatedSalary` and `Balance`
- Salary and balance were converted to numeric values
- `HasCrCard` and `IsActiveMember` were converted from `Yes/No` to `1/0`

### Geography Standardization

Inconsistent geography values such as:

- `French`
- `FRA`

were standardized to:

- `France`

### Dataset Validation

After cleaning:

- Missing values: **0**
- Duplicate rows: **0**
- Duplicate Customer IDs: **0**
- Final records: **10,000**

---

## Exploratory Data Analysis

Initial analysis was performed to understand the distribution of:

- Credit Score
- Age
- Tenure
- Number of Products
- Customer Churn

The overall churn rate in the cleaned dataset is:

**20.37%**

---

## Key Findings

Several patterns were observed during the analysis.

### Geography

Observed churn rates:

| Geography | Churn Rate |
|---|---:|
| France | 16.15% |
| Germany | 32.44% |
| Spain | 16.67% |

### Gender

| Gender | Churn Rate |
|---|---:|
| Female | 25.07% |
| Male | 16.46% |

### Activity Status

| Active Member | Churn Rate |
|---|---:|
| No | 26.85% |
| Yes | 14.27% |

### Age Groups

| Age Group | Churn Rate |
|---|---:|
| 18–30 | 7.52% |
| 31–40 | 12.08% |
| 41–50 | 33.98% |
| 51–60 | 56.21% |
| 61+ | 24.78% |

### Number of Products

| Products | Churn Rate |
|---|---:|
| 1 | 27.71% |
| 2 | 7.58% |
| 3 | 82.71% |
| 4 | 100.00% |

The 3- and 4-product groups are much smaller than the 1- and 2-product groups, so these results should be interpreted in the context of their group sizes.

These findings represent **observed associations in this dataset and do not establish causation**.

---

## Power BI Dashboard

The Power BI report contains three main pages.

### Page 1 — Churn Overview

Provides a high-level view of:

- Total Customers
- Churned Customers
- Stayed Customers
- Churn Rate
- Stayed Rate
- Churn by Geography
- Churn by Age Group
- Churn by Activity
- Churn by Number of Products

### Page 2 — Deeper Analysis

Focuses on:

- Geography filtering
- Average Age
- Average Balance
- Average Credit Score
- Churn by Balance Group
- Age × Balance analysis
- Age × Geography analysis
- Geography-specific churn rate

### Page 3 — Customer Risk Segmentation

Explores customer segments using:

- Age Group
- Activity status
- Geography
- Number of Products
- Churned customer counts
- Cross-segment churn rates

---

## Project Structure

```text
CustomerChurn/
│
├── data/
│   ├── raw/
│   │   └── Bank_Churn_Messy.xlsx
│   │
│   └── processed/
│       └── customer_churn_cleaned.csv
│
├── src/
│   └── data_pipeline.py
│
├── outputs/
│   └── charts/
│
├── .gitignore
├── README.md
└── requirements.txt
