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
Since it was difficult to find a slot that meets everyone's schedule, we resorted to catching-up live during class sessions and exchanging messages in our Slack channel. GitHub is used for code version control. Saturday tutorial session will also be used to keep track on group progress when members are able to attend and work on certain tasks.<br>
<br>
Link to Group 10's public repository: https://github.com/XDarkPhoenixes/Group_10

## Toolbox:
 - Python 3 (sklearn, plotly, pandas, pycaret, numpy)
 - Jupyter Notebook
 - Google Colab
 - GitHub
 - Slack
 - Tableau

## Roles:
 - Communication strategists: Gabriela, Tron
 - Project description: Sarah, Joshua, Teodora
 - Database management: Joshua
 - Modelling: Gabriela, Tron
 - Model comparison with pycaret, model tuning: Teodora
 - Tableau Visualizations:  Sarah
 - Editing, proofreading, troubleshooting: ALL

## Description of Storyboard
[click here for the storyboard](https://public.tableau.com/shared/HTPY7BKPR?:display_count=n&:origin=viz_share_link)

Why Tableau: 
Tableau was used to create the data visualizations and storyboard. We used this tool because it allowed us to easily handle large amounts of data while switching between different visulization types. This quick transformation process allowed us to efficiently find the model that best displayed the message we wanted to convey. Additionally, Tableau allows easy conversion between data types which was necessary for converting the binary categorical variables in our dataset from numerical to string values. Furthermore, the visualizations are interactive which allow users to easily explore the complex data display. 

How to Interact with the Visualizations:
There are a few ways to interact with the data visualization in this analysis. You can hover over the elements of the display (ex. slice of the pie chart, bar of the bar chart, data point on the scatter plot etc.) to view specific information on the count values and dimensions associated with that element. Another method to interact with the display is through filters. For the 'Customer Churn by Tenure, Charges, and Contract type' scatterplot at the end of the story board, you can select or deselect the churn filters to view a more specific visualization. To only view customers that churned, deselect 'No Churn'. In contrast, to only view customers that did not churn, deselect 'Yes Churn'.

## Description of the Analysis Phase of the Project
(...)

## Machine Learning Model 

### Description of preliminary data preprocessing
(....)

### Description of Preliminary Feature Engineering and Preliminary Feature Selection
A library called Pycaret was used to evaluate the feature importance and determine which features to use. The code is available in [this Google Colab Notebook](https://github.com/XDarkPhoenixes/Group_10/blob/main/PycaretTEST.ipynb). The following plot ranks the features of the dataset according to their importance:
![image](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/featuresselection.png) <br>

Since the original dataset wasn't rich in features, it was decided to use all available columns when creating our churn prediction model. As the project expands into more complex production-level datasets, it can be expected that unnecessary columns might exist. For the current phase of our project, though, no features were left out.

### Description of How Data Was Split into Training and Testing Sets
We used the train_test_split function to split the data into testing and training sets, example code follows: <br>
```
# Split our preprocessed data into out features and target arrays
y = Telco_df["Churn_Yes"].values
X = Telco_df.drop(['Churn_Yes','Churn_No'],1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```

###  Explanation of Model Choice, Limitations and Benefits
A library called pycaret was used for model comparison and selection. fAfter going though the initial preprocessing and data splitting steps, we produced the following model comparison with Pycaret:
![models](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/model%20comparison.JPG)

With our project, we are aiming to correctly identify as many churning customers as possible. Therefore, the metric that is most important for us in the Recall. As seen in the chart above, the model giving the best Recall result is Naive Bayes. <br>

Even though we went through feature engineering tests with the purpose of selecting a business-smart model that takes into account an imaginary lifetime value and promo code value, at this stage we decided not to overcomplicate the model selection and proceed with Naive Bayes as our model of choice. <br>

Tuning was done in order to further improve the performance of the model. <br>

#### Benefits of Naive Bayes
 - Comparatively simple model that assumes a Gaussian distribution of the data. 
 - Best recall, which means this model is able to correctly flag the largest amount of churning customers.

#### Limitations
 - The model's accuracy is not as good as that of other models.
 - Tuning is not easy to do with this model, as only the 'var_smoothing' parameter can be evaluated. 
 - Depending of the business goal and retention campaign that executes according to the model predictions, Naive Bayes may not be the best choise from a business perspective. 








