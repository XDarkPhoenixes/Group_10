# Predicting Customer Churn

Data Analytics Boot Camp 6/2021 - UofT SCS

[Group Presentation (Google Slides)](https://docs.google.com/presentation/d/1z_FINC6oIh9rfipcNyjEfOnYt6zm2i0fSWdWJcuA-I4/edit?usp=sharing)

A project by Group 10: <br>
[Gabriela Tuma](https://www.linkedin.com/in/gabrielatuma/) <br>
[Joshua (Jiaxin) Hao](https://github.com/hjx2019) <br>
[Sarah Mir](https://github.com/Smir3) <br>
[Teodora Zlatanova-Geroeva](https://www.linkedin.com/in/teodora-zlatanova-geroeva-5a47aa20/) <br>
[Tron Zi Heng Zhou](https://www.linkedin.com/in/zi-heng-tron-zhou-690722168/) <br>

# Project Overview 

When given the task to find a dataset and develop an analysis to tell a story, the group decided to prioritize the real life, on the job, experience that the project can provide. Since one of the members currently works for a software company with a subscription business model, the dataset chosen was from an organization with similar operations. The group believes that studying the similar data can bring great insights about churn analysis and provide valuable knowledge for future projects. 

Even though the project eventually finds a model that yields the best results for the Telco dataset, it doesn't mean that all churn analysis should be done using the same method, it's important to remember that different datasets interact with models distinctively. 

In order to follow the modelling process without coming across errors two files were created to summarize the results found during the first two deliverables: one retrieving data from postgres and one using the csv file saved locally.

[CSV_Telco_Customer_Churn.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/d88525b5a6c1cd62938491d7af8db76be536b1e6/CSV_Telco_Customer_Churn.ipynb)

[Summary_Modelling_Database.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/d88525b5a6c1cd62938491d7af8db76be536b1e6/Summary_Modelling_Database.ipynb)

## Customer Churn

Customer retention is an important KPI for companies with a subscription-based business model. Customer churn is defined as the percentage of customers that stopped using your company’s product or service during a certain time frame. <br>

The ability to predict customer churn in advance allows companies to retain customers at the highest risk of churn by proactively engaging with them. This represents a huge potential for additional revenue.<br>

![Customer Churn Prediction - example workflow](https://github.com/XDarkPhoenixes/Group_10/blob/8cb2b7f2688183544c5fee3393be8f8ea91e4ca6/Resources/graph.png)
[Image source](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)

Shrinking customer churn as much as possible is a challenging task that all SaaS (Software as a service) organizations are dealing with. Machine Learning models can be utilized to produce a prediction model that can help organizations pinpoint customers at risk of churning. After identifying a cohort of customers who are likely to churn, organizations can test campaigns and strategies for customer retention - discount offerings, frequent check-ins, customer success initiatives, etc. <br>

A customer churn predictive model predicts the churn within a given time interval. This time interval depends on the use-case; it could be one month, three months, or even six months in advance. The cut-off date for the interval must be carefully determined and no information after the cut-off date should be used in the machine learning model.<br>

[Article Predict Customer Churn (the right way) using PyCaret by Towards Data Science](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac) inspired this project.


## Data source

The selected datasource is a famous Telecom Customer Churn dataset from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

Each row represents a customer, each column contains customer’s [attributes described](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) on the column Metadata. The data set includes information about:

 - Customers who left within the last month – the column is called Churn
 - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
 - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
 - Demographic info about customers – gender, age range, and if they have partners and dependents

[customer_churn_data.csv source file](https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv) <br>

## Questions we hope to answer during the project

 - What is the ML model that is most likely to accurately predict customer churn?
 - Are there any business-smart metrics that need to be taken into consideration?
 - What is the best cut-off date for the prediction?
 - Are the available fields in the data source enough to come up with a good model?
 - Does the data source need any cleaning before it can be fed into a model?
 - Optional: Strategies to communicate the results with non-technical senior level audience.

# Results

## Data preprocessing


The Telco dataset retrieved from Kaggle was pretty much clean and didn't require extensive preprocessing. The original table has 7043 rows, after dropping the null values the number went down to 7032, only 11 rows were eliminated, a small percentage that doesn't need further investigation. Also, it was necessary to eliminate the customer ID column and adjust the data type of 'seniorcitizen' and 'tenure'. 

 Data types after adjustments:
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Resources/Preprocessing%20data%20type.png">
</kbd>  &nbsp;
</p>


A lot of the methods only accept numerical inputs, the dataset was encoded using **OneHotEncoder()** so that categorical variables could be properly (and safely) processed by all the models. 

Categorical columns:
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Resources/Categorical%20Variables.png%20.png">
</kbd>  &nbsp;
</p>


Also, two other tools, **train_test_split()** and **StandardScaler()**, were used to standardize the input data. Datasets will often contain features highly varying in magnitudes, units and range, scaling is necessary to guarantee that those numbers are not going to be misleading to the analysis. If not correctly scaled, the algorithm might consider 100m a larger value than 1km for example. 

The same preprocessing steps were used throughout the project in order to fairly compare models. A library called Pycaret was being tested and was used before and after data preprocessing, serving in different parts of the project, such as model choice, data exploration and analysis. 


### Preliminary Feature Engineering and Preliminary Feature Selection

The Pycaret library was used to initially evaluate the feature importance and determine which features to use. The code is available in [this Google Colab Notebook](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/PycaretTEST.ipynb). The following plot ranks the features of the dataset according to their importance:
![image](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/featuresselection.png) <br>

Then, [RandomForestClassifier](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/Telco_churn_RandomForestClassifier.ipynb) was used to create the below feature importance view:
```
# List features with more impact 

important_features = pd.Series(data=rf_model.feature_importances_,index=X.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features.head(15)
TotalCharges                      0.160069
tenure                            0.142756
MonthlyCharges                    0.137544
Contract_Month-to-month           0.048294
OnlineSecurity_No                 0.033924
PaymentMethod_Electronic check    0.031771
TechSupport_No                    0.030719
InternetService_Fiber optic       0.024657
SeniorCitizen                     0.019848
Contract_Two year                 0.017867
gender_Female                     0.017590
gender_Male                       0.017520
OnlineBackup_No                   0.017464
PaperlessBilling_No               0.015909
PaperlessBilling_Yes              0.015661
dtype: float64
```
<br>
Since the original dataset wasn't rich in features, it was decided to use all available columns when creating our churn prediction model. As the project expands into more complex production-level datasets, it can be expected that unnecessary columns might exist. For the current phase of our project, though, no features were left out.

### How Data Was Split into Training and Testing Sets

Given the project's goal to predict customer churn, the Churn_Yes column was defined as the dependent variable (y) and the train_test_split() function was used to split the data into testing and training sets: <br>
```
# Split our preprocessed data into out features and target arrays
y = Telco_df["Churn_Yes"].values
X = Telco_df.drop(['Churn_Yes','Churn_No'],1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```



## Model Choice, Limitations and Benefits

Pycaret was also used for model comparison and selection. After going though the initial preprocessing and data splitting steps, we produced the following model comparison:
![models](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/model%20comparison.JPG)

In this project, we want to correctly identify as many churning customers as possible, therefore, the metric that is most important for us in the Recall. As seen in the chart above, the model giving the best Recall result is Naive Bayes. <br>

Even though we went through feature engineering tests with the purpose of selecting a business-smart model that takes into account an imaginary lifetime value and promo code value, at this stage we decided not to overcomplicate the model selection and proceed with Naive Bayes as our model of choice. <br>

Tuning was done in order to further improve the performance of the model. <br>

#### Benefits of the Naive Bayes classifier
 - Comparatively simple model that assumes a Gaussian distribution of the data.
 - Based on Bayes' Theorem with an assumption of independence among predictors. 
 - Best recall, which means this model is able to correctly flag the largest amount of churning customers.

#### Limitations
 - The model's accuracy is not as good as that of other models.
 - Tuning is not easy to do with this model, as only the 'var_smoothing' parameter can be evaluated. 
 - Depending of the business goal and retention campaign that executes according to the model predictions, Naive Bayes may not be the best choice from a business perspective. 

## Database Integration

> To the grader: our data comes from a single resource, and the data is already Normalized. In this situation, we use one table to save the data, and we don't need to join table with the database.

### Cloud-Based DataBase with AWS:

* The team created a postgresSQL DB on AWS

![image](https://user-images.githubusercontent.com/48306359/142778971-0aef89f4-073b-4352-8298-654cd6a7c561.png)



* pgAdmin is the plat-form used to run sql script [create_table.sql](create_table.sql), like creating the table; load data; and monitor performances

![image](https://user-images.githubusercontent.com/48306359/142779001-9c5ee767-9dcf-4194-b182-fa7fd2216d29.png)


* After the raw data is ready, data are read into google colab with pySpark and Pandas

![image](https://user-images.githubusercontent.com/48306359/142779250-633dc27d-7572-44e8-9f40-42518663b21e.png)



## Data Exploration and Analysis

The dataset was preprocessed and multiple methods were applied to create a churn prediction model. Nevertheless, the data can generate other relevant outcomes for our analysis. The Telco table includes information such as senior, partner, dependents and other features that can be used to better understand the company's customers. 

When using machine learning we can't see clearly the relationships that methods are creating to produce a predictive model, so it's important to explore the dataset in different ways and get insights that might be missed during data processing. 

 

Churned customers represent roughly one quarter of our dataset: 
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Figure%201-%20Number%20of%20Churn.png">
</kbd>  &nbsp;
</p>

The distribution of female and male customers is quite uniform: 
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Figure%202-%20Gender%20Breakdown.png">
</kbd>  &nbsp;
</p>

And the gender doesn't seem to impact other variables of the dataset:
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Figure%204-%20Churn%20by%20Demographics%20and%20Gender.png">
</kbd>  &nbsp;
</p>

Including churn:
<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Figure%206%20-%20Churn%20by%20Charges,%20Contract,%20Tenure%20and%20Gender.png">
</kbd>  &nbsp;
</p>





A very important feature of the churn analysis is the type of contract. Customers with a month-to-month contract tend to cancel their subscription more easily compared to the ones with yearly commitments. 

<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Figure%205-%20Churn%20by%20Charges,%20Tenure,%20and%20Contract.png">
</kbd>  &nbsp;
</p>



## Storyboard - Tableau

Tableau was used to create the data visualizations and storyboard because it allowed us to easily handle large amounts of data while switching between different visualization types. This quick transformation process allowed us to efficiently find the model that best displayed the message we wanted to convey. Additionally, Tableau allows easy conversion between data types which was necessary for converting the binary categorical variables in our dataset from numerical to string values. Furthermore, the visualizations are interactive which allow users to easily explore the complex data display. 

<p align="center">
<kbd>
  <img src="https://github.com/XDarkPhoenixes/Group_10/blob/fc8d3de00619aab0f4d0a1c073cd39ddb3ad2bcf/Tableau%20Visualizations/Story%20Board%20Snapshot.png">
</kbd>  &nbsp;
</p>

There are a few ways to interact with the data visualization in this analysis. You can hover over the elements of the display (ex. slice of the pie chart, bar of the bar chart, data point on the scatter plot etc.) to view specific information on the count values and dimensions associated with that element. Another method to interact with the display is through filters. For the 'Customer Churn by Tenure, Charges, and Contract type' scatterplot at the end of the story board, you can select or deselect the churn filters to view a more specific visualization. To only view customers that churned, deselect 'No Churn'. In contrast, to only view customers that did not churn, deselect 'Yes Churn'.

[Customer Churn storyboard](https://public.tableau.com/shared/HTPY7BKPR?:display_count=n&:origin=viz_share_link)


# Deliverables of the four-segment project

**Presentation**

Content:
- [X] 1. Selected topic
- [X] 1. Reason the topic was selected
- [X] 1. Description of the source of data
- [X] 1. Questions the team hopes to answer with the data
- [X] 2. Description of the data exploration phase of the project
- [X] 2. Description of the analysis phase of the project
- [X] 3. Technologies, languages, tools, and algorithms used throughout the project
- [X] 4. Result of analysis
- [ ] 4. Recommendation for future analysis
- [ ] 4. Anything the team would have done differently

Slides:
- [X] 2. Presentations are drafted in Google Slides
- [X] 3. Slides are primarily images or graphics (rather than primarily text)
- [X] 3. Images are clear, in high-definition, and directly illustrative of subject matter

Live Presentation:
- [ ] 4. All team members present in equal proportions
- [ ] 4. The team demonstrates the dashboard's real-time interactivity
- [ ] 4. The presentation falls within any time limits provided by the instructor
- [ ] 4. The submission includes speaker notes, flashcards, or a video of the presentation rehearsal

Main Branch:
- [X] 1. Description of the communication protocols
- [X] 2. Outline of the project
- [X] 2. All code necessary to perform exploratory analysis
- [X] 3. Description of the communication protocols has been removed
- [X] 4. All code necessary to complete the machine learning portion of the project
- [X] 4. Any images that have been created (at least three)
- [ ] 4. Requirements.txt file

README.md:
- [X] 3. Cohesive, structured outline of the project
- [X] 3. Link to Google Slides presentation - draft

[Group Presentation - draft](https://docs.google.com/presentation/d/1z_FINC6oIh9rfipcNyjEfOnYt6zm2i0fSWdWJcuA-I4/edit?usp=sharing)

- [X] 4. Link to dashboard

[Customer Churn storyboard](https://public.tableau.com/shared/HTPY7BKPR?:display_count=n&:origin=viz_share_link)

- [X] 4. Link to Google Slides presentation

[Group Presentation (Google Slides)](https://docs.google.com/presentation/d/1z_FINC6oIh9rfipcNyjEfOnYt6zm2i0fSWdWJcuA-I4/edit?usp=sharing)

**Machine Learning Model** 

- [X] 2. Description of data preprocessing

[DataBase_Telco_Customer_Churn.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/DataBase_Telco_Customer_Churn.ipynb)

- [X] 2. Description of feature engineering and the feature selection, including the team's decision-making process

[PycaretTEST2.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/9c0e31db78139d114faace798b511321046884f8/Modelling%20and%20Preprocessing/PycaretTEST2.ipynb)

[Telco_churn_RandomForestClassifier.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/Telco_churn_RandomForestClassifier.ipynb)

- [X] 2. Description of how data was split into training and testing sets
- [X] 2. Explanation of model choice, including limitations and benefits

[Summary_Modelling.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/Summary_Modelling.ipynb)

[Naive_Bayes.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/Naive_Bayes.ipynb)

[Naive_Bayes_Tuned.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/f0b873d6eebc12465081b77f7ecf49460b7a3f97/Modelling%20and%20Preprocessing/Naive_Bayes_Tuned.ipynb)

- [X] 3. Explanation of changes in model choice
- [X] 3. Description of how the model was trained

[Summary_Modelling_Database.ipynb](https://github.com/XDarkPhoenixes/Group_10/blob/main/Summary_Modelling_Database.ipynb)

- [ ] 4. Description and explanation of model's confusion matrix, including final accuracy score

**Database Integration**

- [X] 1. Sample data that mimics the expected final database structure or schema
- [X] 1. Draft machine learning module is connected to the provisional database
- [X] 2. Stores static data for use during the project
- [X] 2. Interfaces with the project in some format
- [X] 2. Includes at least two tables
- [X] 2. Includes at least one join using the database language
- [X] 2. Includes at least one connection string

**Dashboard**

- [X] 2. Storyboard on Google Slide(s) 
- [X] 2. Description of the tool(s) that will be used to create final dashboard 
- [X] 3. Images from the initial analysis
- [X] 3. Data (images or report) from the machine learning task
- [X] 3. At least one interactive element

## Roles:
 - Communication strategists: Gabriela, Tron
 - Project description: Sarah, Joshua, Teodora
 - Database management: Joshua
 - Modelling: Gabriela, Tron
 - Model comparison with pycaret, model tuning: Teodora
 - Tableau Visualizations:  Sarah
 - Editing, proofreading, troubleshooting: ALL

## Toolbox:
 - Python 3 (sklearn, plotly, pandas, pycaret, numpy)
 - Jupyter Notebook
 - Google Colab
 - GitHub
 - Slack
 - Tableau
