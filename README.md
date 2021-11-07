# Predicting Customer Churn
A project by Group 10: <br>
[Sarah Mir]() <br>
[Gabriela Tuma](https://www.linkedin.com/in/gabrielatuma/) <br>
[Teodora Zlatanova-Geroeva](https://www.linkedin.com/in/teodora-zlatanova-geroeva-5a47aa20/) <br>
[Tron Zi Heng Zhou](https://www.linkedin.com/in/zi-heng-tron-zhou-690722168/) <br>
[Joshua (Jiaxin) Hao]() <br>

## Purpose:
The ability to predict customer churn in advance allows companies to retain customers at the highest risk of churn by proactively engaging with them. This represents a huge additional potential revenue source for companies.<br>

![Customer Churn Prediction - example workflow](https://github.com/XDarkPhoenixes/Group_10/blob/main/graph.png)
[click here for image source](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)

## Background and Reason for Selecting This Topic:
Customer retention is an important KPI for companies with a subscription-based business model. Customer churn is defined as the percentage of customers that stopped using your company’s product or service during a certain time frame. <br>

Shrinking customer churn as much as possible is a challenging task that all SaaS organizations are dealing with. Machine Learning models can be utilized to produce a prediction model that can help organizations pinpoint customers at risk of churning. After identifying a cohort of customers who are likely to churn, organizations can test campaigns and strategies for customer retention - discount offerings, frequent check-ins, customer success initiatives, etc. <br>

A customer churn predictive model predicts the churn within a given time interval. This time interval depends on the use-case; it could be one month, three months, or even six months in advance. The cut-off date for the interval must be carefully determined and no information after the cut-off date should be used in the machine learning model.<br>

[This step-by-step customer churn prediciton guide](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac) inspired this project. 

## Data source description:
[Link to the source csv file](https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv) <br>
The selected datasource is a famous Telecom Customer Churn dataset from Kaggle. <br>
Each row represents a customer, each column contains customer’s attributes described on the column Metadata. <br>

The data set includes information about:<br>

 - Customers who left within the last month – the column is called Churn
 - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
 - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
 - Demographic info about customers – gender, age range, and if they have partners and dependents

## Questions we hope to answer during the project:
 - What is the ML model that is most likely to accurately predict customer churn?
 - Are there any business-smart metrics that need to be taken into consideration?
 - What is the best cut-off date for the prediction?
 - Are the available fields in the data source enough to come up with a good model?
 - Does the data source need any cleaning before it can be fed into a model?
 - Optional: Strategies to communicate the results with non-technical senior level audience.

## Description of the communication protocols:
 - Slack
 - Video calls
 - GitHub
<br>
Since it was difficult to find a slot that meets everyone's schedule, we resorted to catching-up live during class sessions and exchanging messages in our Slack channel. GitHub is used for code version control.<br>
[Link to Group 10's public repository](https://github.com/XDarkPhoenixes/Group_10)

## Toolbox:
 - Python 3 (sklearn, plotly, pandas, pycaret, numpy)
 - Jupyter Notebook
 - Google Colab
 - GitHub
 - Slack

## Roles:
 - Communication strategists: Gabriela, Tron
 - Project description: Sarah, Joshua, Teodora
 - Modelling: Gabriela, Tron
 - Machine learning model comparison with pycaret: Teodora
 - Editing, proofreading, troubleshooting: ALL
