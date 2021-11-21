# Predicting Customer Churn

Data Analytics Boot Camp 6/2021 - UofT SCS

A project by Group 10: <br>
[Gabriela Tuma](https://www.linkedin.com/in/gabrielatuma/) <br>
[Joshua (Jiaxin) Hao](https://github.com/hjx2019) <br>
[Sarah Mir](https://github.com/Smir3) <br>
[Teodora Zlatanova-Geroeva](https://www.linkedin.com/in/teodora-zlatanova-geroeva-5a47aa20/) <br>
[Tron Zi Heng Zhou](https://www.linkedin.com/in/zi-heng-tron-zhou-690722168/) <br>

# Project Overview 

When given the task to find a dataset and develop an analysis to tell a story, the group decided to prioritize the real life, on the job, experience that the project can provide. Since one of the members currently works for a software company with a subscription business model, the dataset chosen was from an organization with similar operations. The group believes that studying the similar data can bring great insights about churn analysis and provide valuable knowledge for future projects. 

Even though the project eventually finds a model that yields the best results for the Telco dataset, it doesn't mean that all churn analysis shoud be done using the same method, it's important to remember that different datasets interact with models distinctively. 


## Customer Churn

Customer retention is an important KPI for companies with a subscription-based business model. Customer churn is defined as the percentage of customers that stopped using your company’s product or service during a certain time frame. <br>

The ability to predict customer churn in advance allows companies to retain customers at the highest risk of churn by proactively engaging with them. This represents a huge potential for additional revenue.<br>

![Customer Churn Prediction - example workflow](https://github.com/XDarkPhoenixes/Group_10/blob/main/graph.png)
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

################################

### Preliminary Feature Engineering and Preliminary Feature Selection

A library called Pycaret was used to initially evaluate the feature importance and determine which features to use. The code is available in [this Google Colab Notebook](https://github.com/XDarkPhoenixes/Group_10/blob/main/PycaretTEST.ipynb). The following plot ranks the features of the dataset according to their importance:
![image](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/featuresselection.png) <br>

Then, RandomForestClassifier was used to create the below feature importance view:
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

We used the train_test_split function to split the data into testing and training sets, example code follows: <br>
```
# Split our preprocessed data into out features and target arrays
y = Telco_df["Churn_Yes"].values
X = Telco_df.drop(['Churn_Yes','Churn_No'],1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```

## Model Choice, Limitations and Benefits

A library called pycaret was used for model comparison and selection. After going though the initial preprocessing and data splitting steps, we produced the following model comparison with Pycaret:
![models](https://github.com/XDarkPhoenixes/Group_10/blob/main/Resources/model%20comparison.JPG)

With our project, we are aiming to correctly identify as many churning customers as possible. Therefore, the metric that is most important for us in the Recall. As seen in the chart above, the model giving the best Recall result is Naive Bayes. <br>

Even though we went through feature engineering tests with the purpose of selecting a business-smart model that takes into account an imaginary lifetime value and promo code value, at this stage we decided not to overcomplicate the model selection and proceed with Naive Bayes as our model of choice. <br>

Tuning was done in order to further improve the performance of the model. <br>

#### Benefits of the Naive Bayes classifier
 - Comparatively simple model that assumes a Gaussian distribution of the data.
 - Based on Bayes' Theorem with an assumption of independence among predictors. 
 - Best recall, which means this model is able to correctly flag the largest amount of churning customers.

#### Limitations
 - The model's accuracy is not as good as that of other models.
 - Tuning is not easy to do with this model, as only the 'var_smoothing' parameter can be evaluated. 
 - Depending of the business goal and retention campaign that executes according to the model predictions, Naive Bayes may not be the best choise from a business perspective. 


###################

## Data Exploration Phase of the Project
(...)

## Analysis Phase of the Project
(...)


## Storyboard - Tableau

Tableau was used to create the data visualizations and storyboard because it allowed us to easily handle large amounts of data while switching between different visulization types. This quick transformation process allowed us to efficiently find the model that best displayed the message we wanted to convey. Additionally, Tableau allows easy conversion between data types which was necessary for converting the binary categorical variables in our dataset from numerical to string values. Furthermore, the visualizations are interactive which allow users to easily explore the complex data display. 

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
- [ ] 3. Technologies, languages, tools, and algorithms used throughout the project
- [ ] 4. Result of analysis
- [ ] 4. Recommendation for future analysis
- [ ] 4. Anything the team would have done differently

Slides:
- [X] 2. Presentations are drafted in Google Slides
- [ ] 3. Slides are primarily images or graphics (rather than primarily text)
- [ ] 3. Images are clear, in high-definition, and directly illustrative of subject matter

Live Presentation:
- [ ] 4. All team members present in equal proportions
- [ ] 4. The team demonstrates the dashboard's real-time interactivity
- [ ] 4. The presentation falls within any time limits provided by the instructor
- [ ] 4. The submission includes speaker notes, flashcards, or a video of the presentation rehearsal

Main Branch:
- [X] 1. Description of the communication protocols
- [X] 2. Outline of the project
- [X] 2. All code necessary to perform exploratory analysis
- [ ] 3. Description of the communication protocols has been removed
- [ ] 4. All code necessary to complete the machine learning portion of the project
- [ ] 4. Any images that have been created (at least three)
- [ ] 4. Requirements.txt file

README.md:
- [ ] 3. Cohesive, structured outline of the project
- [ ] 3. Link to Google Slides presentation - draft
- [ ] 4. Link to dashboard
- [ ] 4. Link to Google Slides presentation

**Machine Learning Model** 

- [X] 2. Description of data preprocessing
- [X] 2. Description of feature engineering and the feature selection, including the team's decision-making process
- [X] 2. Description of how data was split into training and testing sets
- [X] 2. Explanation of model choice, including limitations and benefits
- [ ] 3. Explanation of changes in model choice
- [ ] 3. Description of how the model was trained
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
- [ ] 3. Images from the initial analysis
- [ ] 3. Data (images or report) from the machine learning task
- [ ] 3. At least one interactive element

## Roles:
 - Communication strategists: Gabriela, Tron
 - Project description: Sarah, Joshua, Teodora
 - Database management: Joshua
 - Modelling: Gabriela, Tron
 - Model comparison with pycaret, model tuning: Teodora
 - Tableau Visualizations:  Sarah
 - Editing, proofreading, troubleshooting: ALL


## Communication protocols:
 - Slack
 - Video calls
 - GitHub
<br>
Since it was difficult to find a slot that meets everyone's schedule, we resorted to catching-up live during class sessions and exchanging messages in our Slack channel. GitHub is used for code version control. Saturday tutorial session will also be used to keep track on group progress when members are able to attend and work on certain tasks.<br>
<br>


## Toolbox:
 - Python 3 (sklearn, plotly, pandas, pycaret, numpy)
 - Jupyter Notebook
 - Google Colab
 - GitHub
 - Slack
 - Tableau
