# Data Analysis Projects

---

## 📈 TFE-GHMM — KSE-100 Market Regime Detection

A self-contained Gradio app that uses a Gaussian Hidden Markov Model to detect market regimes (Bull / Sideways / Bear) on the Pakistan Stock Exchange (KSE-100).

### ⬇️ Download

**[Download TFE_GHMM_Gradio.py](https://raw.githubusercontent.com/SUmer22/Projects/main/TFE_GHMM_Gradio.py)**
(Right-click → Save link as… — links are active once this PR is merged to `main`)

### 🚀 Run in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SUmer22/Projects/blob/main/TFE_GHMM_Gradio.py)

Or paste these two cells manually:

```python
# Cell 1 — install dependencies
!pip install gradio pandas numpy matplotlib seaborn yfinance ta hmmlearn scikit-learn scipy -q
```

```python
# Cell 2 — paste the full contents of TFE_GHMM_Gradio.py and run
# A public gradio.live URL will be printed automatically.
```

### 💻 Run Locally

```bash
pip install gradio pandas numpy matplotlib seaborn yfinance ta hmmlearn scikit-learn scipy
python TFE_GHMM_Gradio.py
# → http://localhost:7860
```

### 📋 Required CSV columns

`Date`, `Open`, `High`, `Low`, `Close`, `Volume`
Date format: `YYYY-MM-DD` (comma-formatted numbers are handled automatically)

---

## Customer Churn Analysis

##EDA using Python
1. Data Overview
•	Data Dimensions: The dataset consists of customer information with various features related to their subscription and demographic data.
•	Missing Values: No missing values were found in key columns after handling the TotalCharges column, which was initially filled with blank spaces that were converted to zeros.
•	Data Type Corrections: The SeniorCitizen column was converted to a categorical format with values "Yes" and "No".
2. Churn Overview
•	Churn Distribution: The count plot shows the distribution of churned vs. non-churned customers, with a clear breakdown of how many customers have left versus stayed.
3. Churn by Demographic Features
•	Gender: Men and women churn at similar rates, as indicated by the count plot, where there is no substantial difference in churn based on gender.
•	Partner Status: Customers without a partner have a higher churn rate compared to those with a partner.
•	Dependents: Customers without dependents have a higher churn rate than those with dependents.
•	Senior Citizens: Senior citizens have a noticeably higher churn rate compared to non-senior citizens.
4. Churn by Service Usage
•	Internet Service: Customers with Fiber optic internet service exhibit higher churn rates than those with DSL or no internet service.
•	Streaming TV Service: Churn is higher among customers who have a streaming TV service.
•	TechSupport: Churn rates are higher among customers who do not have tech support services.
•	Online Security: Customers who do not have online security services show higher churn rates. Interestingly, customers without internet service exhibit a relatively lower churn rate in this category.
5. Churn by Contract and Payment Type
•	Contract Type: Customers with month-to-month contracts have a significantly higher churn rate than those with one or two-year contracts.
•	Payment Method: There is a slight tendency for churn to be higher among customers using electronic check payments.
•	Paperless Billing: Customers with paperless billing tend to have a higher churn rate than those who receive paper bills.
6. Financial Factors Impacting Churn
•	Monthly Charges: There is a clear pattern where customers who churn tend to have higher monthly charges compared to those who stay.
•	Total Charges: Similarly, churned customers also show higher total charges, which might indicate that these customers have been paying more over time without feeling sufficiently satisfied.
7. Tenure and Churn
•	Tenure Distribution: Customers who have been with the company for shorter periods (0-12 months) tend to churn at a higher rate compared to long-term customers (60+ months).
•	Tenure Group Analysis: The analysis also revealed that the churn rate is highest in the early months of tenure (0-12 months) and gradually decreases as the customer stays longer with the company.
8. Multi-Factor Analysis
•	Churn by Monthly Charges and Contract Type: Customers with higher monthly charges and month-to-month contracts have the highest churn rates.
•	Tenure and Contract Type: Month-to-month contracts combined with shorter tenures are the primary drivers of customer churn.
Key Insights for Actionable Strategies:
1.	Retention Focus for New Customers: A key segment for churn reduction is customers with low tenure, especially those in month-to-month contracts. Providing incentives to renew or extend contracts could help retain them.
2.	Enhance Support Services: Offering additional services like tech support and online security can reduce churn rates, as these add-ons appear to correlate with longer customer retention.
3.	Evaluate Pricing Strategy: Customers with higher charges, particularly those with short tenure or no contract, are more likely to churn. Reviewing pricing tiers or offering discounts for long-term contracts might help reduce churn.
4.	Improve Engagement with Senior Citizens: The higher churn rate among senior citizens suggests that additional support or specialized plans for this demographic could improve retention.

