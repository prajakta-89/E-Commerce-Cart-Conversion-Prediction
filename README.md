# 🛒 E-Commerce-Cart-Conversion-Prediction
An end-to-end Machine Learning project that predicts whether a customer will complete a purchase after adding products to their shopping cart. The project combines customer behavior, website activity, product information, order history, and customer service data to help e-commerce businesses reduce cart abandonment and improve conversion rates.

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dashboard Preview](#dashboard-preview)
- [Solution](#solution)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Exploratory Data Analysis & Visualizations](#-exploratory-data-analysis--visualizations)
- [Dataset Information](#dataset-information)
- [Data Preprocessing](#data-preprocessing)
- [Feature Importance](#feature-importance)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Technology Used](#technology-used)
- [Installation](#installation)
- [Business Insights](#business-insights)
- [Business Impact](#business-impact)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

## Project Overview

Cart abandonment is one of the biggest challenges faced by e-commerce businesses, resulting in significant revenue loss. This project develops a predictive machine learning model to identify customers who are likely to abandon their carts before completing a purchase.

The solution enables businesses to take proactive actions such as personalized offers, reminder emails, and targeted marketing campaigns to increase conversion rates and recover lost revenue.


## Business Problem

ShopSphere, a large e-commerce retailer, observed that a high percentage of customers add products to their shopping carts but leave the website without completing their purchases.

This behavior results in:

- Revenue loss
- Low conversion rates
- Inefficient marketing campaigns
- Poor customer engagement

The company requires a predictive analytics solution capable of identifying customers who are likely to abandon their carts so that timely interventions can be applied.

## Dashboard Preview
### Streamlit Dashboard
<p>
  <img src="Images/Dashboard1.png" width="800"/>
</p>



## Solution
A Machine Learning Cart Conversion Prediction System was developed to predict whether a customer will complete a purchase after adding products to the cart.

### Target Variable

- **0 → Cart Abandonment**
- **1 → Purchase Completion**

The model analyzes customer behavior, website interactions, purchase history, product information, and satisfaction metrics to estimate purchase probability.

### Streamlit Dashboard
<p>
  <img src="Images/Dashboard_2.png" width="800"/>
</p>


## Project Structure

```
Cart-Conversion-Prediction/
│
├── Datasets/
│   ├── Website_Activity.csv
│   ├── Customer_Department_Targets.csv
│   ├── Products.csv
│   ├── Orders.csv
│   └── Customer_Service.csv
│
├── notebooks/
│   └── E_Commerce_Cart_Conversion_Prediction.ipynb
│
├── Model_Files/
│   ├── threshold.pkl
│   ├── features.pkl
│   └── rf_model.pkl
│
├── Streamlit_Dashboard/
│   └── app.py
│
├── Images/
│   ├── Dashboard1.png
│   ├── Important_Features
│   ├── Roc_Curve
│   ├── Hist_Plot
│   ├── Heatmap
│   ├── Target_Distribution
│   └── Dashboard_2.png
│
├── Requirements.txt
│
└── README.md
```

## Machine Learning Models
The following models were trained and evaluated:
- Random Forest + Stratify
- Random Forest + Smote
- Logistic Regression
- Decision Tree
- XGBoost


## Model Evaluation
Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## 📊 Exploratory Data Analysis & Visualizations
The following visualizations provide insights into customer behavior, purchase patterns, and the key factors influencing cart conversion.


##  1. Target Distribution
<p>
  <img src="Images/Target_Distribution.png" width="500"/>
</p>

- This chart shows the distribution of the target variable **Purchase_After_Cart**.

- **0 → Cart Abandonment**
- **1 → Purchase Completion**

### Key Insight

- Approximately **83%** of customers abandoned their carts.
- Only **17%** completed their purchase.

This severe class imbalance justified the use of **SMOTE** during model training.


##  2. ROC Curve
<p>
  <img src="Images/Roc_Curve.png" width="500"/>
</p>

- The ROC Curve evaluates the model's ability to distinguish between customers who will purchase and those who will abandon their carts.

### Key Insight

- A curve closer to the top-left corner indicates better model performance.
- ROC-AUC provides a threshold-independent measure of classification quality.
- Higher AUC values indicate stronger predictive capability.


##  3. Feature Distribution
<p>
  <img src="Images/Hist_Plot.png" width="700"/>
</p>

Histograms display the distribution of important numerical features such as:

- Pages Viewed
- Session Duration
- Customer Lifetime Value
- Average Order Value
- Product Rating

### Key Insight

The distributions reveal customer browsing and purchasing patterns, helping identify skewed variables and understand customer engagement before model training.



##  4. Correlation Heatmap
<p>
  <img src="Images/Heatmap.png" width="700"/>
</p>

- The heatmap illustrates the correlation between numerical features in the dataset.

### Key Insight

- Strong positive relationships indicate features that increase purchase probability.
- Weakly correlated variables provide unique information to the model.
- The heatmap also helps identify multicollinearity among features.


## Dataset Information

The Cart Conversion Prediction project uses multiple datasets representing different aspects of an e-commerce business. These datasets were integrated to build a comprehensive view of customer behavior and improve the accuracy of purchase prediction.

The final analytical dataset was created by merging customer information, website activity, product details, order history, and customer service records using unique identifiers.

### Dataset Summary

Dataset
- Website_Activity 
- Customer_Department_Targets
- Customer_Service
- Orders
- Products



## 1. Website Activity Dataset
### Purpose

This dataset captures how customers interact with the e-commerce website before making a purchase decision. It provides behavioral information that helps understand customer engagement and browsing patterns.

### Features
| Feature | Description |
|----------|-------------|
| Customer_ID | Unique identifier for each customer |
| Device_Type | Device used to browse the website (Desktop, Tablet, iOS, etc.) |
| Traffic_Source | Source through which the customer visited the website (Organic, Social, Referral, Paid Search) |
| Pages_Viewed | Total number of pages visited during the session |
| Session_Duration | Total time spent on the website |
| Added_To_Cart | Indicates whether products were added to the shopping cart |
| Purchase_After_Cart | Target variable (0 = Cart Abandonment, 1 = Purchase Completion) |

### Business Importance
Website activity provides valuable insights into customer engagement. Customers who browse more pages and spend more time on the website are generally more likely to complete a purchase.



## 2. Customer Department Targets Dataset
### Purpose

This dataset contains customer profile information, loyalty details, purchasing history, and marketing engagement metrics.

### Features

| Feature | Description |
|----------|-------------|
| Customer_ID | Unique customer identifier |
| Loyalty_Tier | Bronze, Silver, Gold, or Platinum membership |
| Tenure_Months | Number of months the customer has been associated with the company |
| Last_Login_Days_Ago | Days since the customer's last login |
| Total_Orders | Total number of historical orders |
| Avg_Order_Value | Average value of customer orders |
| Recency_Days | Number of days since the last purchase |
| Customer_Lifetime_Value | Total revenue generated by the customer |
| Campaign_Click_Rate | Percentage of marketing campaign clicks |
| Avg_Satisfaction | Average customer satisfaction score |

### Business Importance

Customer profile information helps identify loyal, high-value customers and measures long-term purchasing behavior. These variables are essential for understanding customer retention and future purchase potential.


## 3. Customer Service Dataset

### Purpose

This dataset records customer interactions with support services and overall satisfaction.

### Features

| Feature | Description |
|----------|-------------|
| Customer_ID | Unique customer identifier |
| Satisfaction_Score | Customer satisfaction rating after service interactions |

### Feature Engineering

The average satisfaction score was calculated for each customer.

Generated Feature:

```
Avg_Service_Satisfaction
```

### Business Importance

Customer satisfaction has a direct influence on repeat purchases. Customers with higher satisfaction scores generally demonstrate higher purchase conversion rates.


## 4. Orders Dataset

### Purpose

The Orders dataset stores historical purchase information for customers.

### Features

| Feature | Description |
|----------|-------------|
| Order_ID | Unique order identifier |
| Customer_ID | Customer identifier |
| Product_ID | Purchased product identifier |
| Quantity | Number of products purchased |

### Business Importance

Order history helps understand purchasing frequency and customer buying behavior. It also enables linking customers with product information.



## 5. Products Dataset

### Purpose

This dataset contains information about the products available on the platform.

### Features

| Feature | Description |
|----------|-------------|
| Product_ID | Unique product identifier |
| Product_Rating | Average customer rating |
| Product_Price | Selling price |
| Return_Rate | Percentage of returned products |

### Feature Engineering

Average values were calculated for each customer.

Generated Features

- Avg_Product_Rating
- Avg_Product_Price
- Avg_Return_Rate

### Business Importance

Product quality and pricing strongly influence purchase decisions. Customers tend to purchase products with higher ratings and lower return rates.

Datasets were merged using:

- Customer_ID
- Product_ID

This integration ensured that every customer record contained website behavior, purchasing history, product characteristics, and customer satisfaction metrics.

## Dataset Characteristics

- **Total Customer Sessions:** 300,000
- **Multiple Data Sources:** 5
- **Merged into a Unified Dataset:** Yes
- **Categorical Features:** One-Hot Encoded
- **Numerical Features:** Standard business and behavioral metrics
- **Missing Values:** Handled during preprocessing
- **Class Imbalance:** Addressed using SMOTE
- **Train-Test Split:** 80:20 using Stratified Sampling
 

## Data Preprocessing
The following preprocessing steps were performed:

- Removed unnecessary columns
- Handled missing values
- Aggregated customer service metrics
- Aggregated product information
- Merged multiple datasets
- One-Hot Encoding
- Feature Engineering
- Class imbalance handling using SMOTE
- Train-Test Split using Stratified Sampling

## Feature Importance

<p>
  <img src="Images/Important_Features.png" width="700"/>
</p>

### Key Insight
Feature importance identifies the variables that contribute most to predicting purchase completion.
Customer engagement, shopping behavior, and product-related attributes have the strongest influence on conversion.



## Exploratory Data Analysis

Key analyses performed:

- Cart Abandonment Distribution
- Device Type Analysis
- Traffic Source Analysis
- Loyalty Tier Analysis
- Customer Lifetime Value Analysis
- Product Rating Analysis
- Session Duration Analysis
- Pages Viewed Analysis
- Feature Importance



## Technology Used

- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib

### Visualization
- Matplotlib
- Seaborn
- Streamlit


## Installation
Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```


## Business Insights
Key findings include:

- Most customers abandon their carts before checkout.
- Session duration positively influences purchase completion.
- Customers with higher loyalty tiers have better conversion rates.
- Customer Lifetime Value is strongly associated with purchase probability.
- Marketing engagement significantly impacts conversion.
- Product ratings and pricing influence customer purchase decisions.


## Business Impact
The developed solution enables the company to:

- Predict customers likely to abandon their carts.
- Launch personalized remarketing campaigns.
- Send targeted discount offers.
- Recover lost revenue.
- Improve marketing ROI.
- Increase checkout completion rate.
- Enhance customer experience.
 
## Future Enhancements
- Deep Learning Model
- Real-Time Prediction API
- Customer Segmentation
- Recommendation System Integration
- Automated Marketing Campaign Trigger
- Cloud Deployment (AWS/Azure)


## Author
**Prajakta Bhondave**
Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning | Streamlit

LinkedIn: https://www.linkedin.com/in/prajakta-bhondave-773b092a6/

GitHub: https://github.com/prajakta-89
