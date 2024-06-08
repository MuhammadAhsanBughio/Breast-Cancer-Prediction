#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center"> <h1> BREAST CANCER DETECTION </h1></div>

# In[296]:


#Importing Important Libraries

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Algorithms
#Random Forest
from sklearn.ensemble import RandomForestClassifier
#K Neighbor
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn.svm import SVC
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
#Logistic Regression
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings("ignore")


# #### Loading and displaying data

# In[297]:


MyData= pd.read_csv("breast-cancer.csv")
MyData.head(4)


# ## INTRODUCTION

# Breast cancer is a global health issue that impacts numerous individuals and their loved ones, emphasizing the need for early detection and precise diagnosis to enhance treatment effectiveness and patient outcomes. Over the past few years, the application of machine learning methods has demonstrated considerable potential in supporting healthcare practitioners with breast cancer diagnosis, offering a promising avenue for improved medical decision-making.
# 
# This project aims to analyze different features extracted from breast cancer patients in order to develop an accurate and efficient breast cancer prediction model. We will be using Supervised learning models which utilizes labeled data to learn patterns and make predictions. By training these models on a dataset of known breast cancer cases, we can develop algorithms that can accurately classify new, unseen instances as either Malignant (cancerous) or Benign (non-cancerous)
# 
# The primary aim of this project is to assess and compare the effectiveness of various supervised learning algorithms, including logistic regression, support vector machines, decision trees, random forests, SVM, and Naive Bayes. The evaluation will be based on metrics such as accuracy, precision, and recall. The results obtained from this study carry substantial importance in the realm of breast cancer diagnosis, potentially impacting the advancement of diagnostic methods in a meaningful way.

# ### DATASET EXPLAINED

# The kaggle dataset for breast cancer analysis consists of 569 rows and 32 columns. It consists of various features that are relevant to diagnosing breast cancer. Each data entry is identified by a unique "id" and is accompanied by a "diagnosis" label (target variable), indicating whether the breast mass is classified as Malignant (cancerous) or Benign (non-cancerous). The dataset encompasses informative features such as "radius_mean," which represents the average distance from the center to points on the breast mass's perimeter, providing insights into its size. "Texture_mean" signifies the variations in gray levels within the mass, serving as an indicator of textural characteristics. "Perimeter_mean" denotes the boundary length around the tumor, reflecting its extent. The "area_mean" feature measures the size of the mass in terms of the number of pixels, potentially indicating the presence of malignancy for larger areas.
# 
# "Smoothness_mean" characterizes the local variation in radius lengths, providing an average measure of smoothness within the mass. "Compactness_mean" quantifies how closely the points on the mass are packed together, conveying its compactness. "Concavity_mean" reflects the severity of concave regions within the mass, while "concave points_mean" signifies the number of such concave portions. The dataset also includes corresponding features for the "worst" and "standard error" values of the mean features mentioned above. For example, "radius_worst" represents the largest observed radius value within the breast mass, while "radius_se" represents the standard error associated with the radius measurement.
# 
# In addition, the dataset incorporates features related to smoothness, compactness, concavity, and concave points in their worst and standard error variations, further enriching the available information. Other relevant features like symmetry and fractal dimension are also present in the dataset, providing insights into the symmetry and complexity of the breast mass.
# 
# By analyzing these features and their variations, valuable insights can be gained regarding the characteristics of breast masses and their potential correlation with malignant or benign classifications.

# ### PRELIMINARY ANALYSIS

# ### Pre Processing

# In[298]:


MyData.shape


# In[299]:


print(MyData.info())


# The datatypes are fine except Diagnosis target variable which we will be changing to binary 1 and 0 later 

# In[300]:


#checking Null Values
MyData.isna().sum()


# There is no null value in the dataset

# In[301]:


#Unique rows in each feature
MyData.nunique()


# Diagnosis column has 2 unique values, indicating the two categories used to classify breast cancer instances, Malignant and Benign. Other columnns has warying numbers of unique values, reflecting the diversity and range of measurements or features extracted from breast cancer data.

# In[302]:


#checking for duplicates
MyData.duplicated().sum()


# There are no duplicates in the dataset

# In[303]:


#dropping Id column as we don't need it in our analysis
MyData = MyData.drop("id", axis = 1)
MyData.head()


# ### Data Exploratory Analysis

# In[304]:


# Desriptive Analysis
MyData

MyData.describe()


# #### Observations:

# One notable observation is that the mean radius of the breast masses in the dataset is approximately 14.13 units, with a range spanning from 6.98 to 28.11. This suggests significant variation in the size of the masses, with some instances exhibiting considerably larger or smaller radii. 
# 
# Another intriguing finding is related to the mean area of the breast masses, which is approximately 654.89 square units, with a wide range from 143.50 to 2501.00. This indicates a substantial diversity in the size and extent of the masses, ranging from relatively small to significantly large areas.
# 
# Furthermore, the average smoothness of the breast masses is approximately 0.096, with a minimum value of 0.0526 and a maximum value of 0.1634. This finding suggests that the texture and surface characteristics of the masses can vary considerably, with some exhibiting smoother surfaces and others displaying rougher textures.
# 
# Additionally, the mean concavity of the breast masses is approximately 0.0888, with a maximum value of 0.4268. This indicates that while some masses exhibit minimal concave regions, others can have significant concavity, potentially indicating more complex and irregular shapes.
# 
# Lastly, the mean fractal dimension of the breast masses is approximately 0.0628, with values ranging from 0.04996 to 0.09744. This finding suggests that the complexity and intricacy of the masses' geometric patterns can vary, with some instances displaying more intricate and self-similar structures.
# 
# These interesting findings highlight the diversity and variability within the dataset, emphasizing the importance of considering multiple features when analyzing and predicting breast cancer. The variations in size, area, texture, concavity, and fractal dimension provide valuable insights for further exploration and potential correlations with the diagnosis of breast cancer.

# #### Distribution of Diagnosis Feature

# In[305]:


print('Count of Malignant or Benign cells in diagnosis')
MyData['diagnosis'].value_counts()


# In[306]:


# Get the value counts of the "diagnosis" column
diagnosis_counts = MyData["diagnosis"].value_counts()

# Plotting the pie chart
plt.figure(figsize=(4, 4))
plt.pie(diagnosis_counts, labels=diagnosis_counts.index, autopct='%1.1f%%', colors=["green", "red"])
plt.title("Distribution of Diagnosis")
plt.show()


# There are 62.7% Benign (Noncancerous) cells and  37.3% Malignant (Cancerous) cells in the dataset.

# #### Histogram for the Numerical Features

# In[307]:


# Defining the number of rows and columns for the grid layout
nrows = 5
ncolumns = (len(Data.columns) - 1) // nrows

# Creating a grid layout for the feature distribution plots
fig, axes = plt.subplots(nrows, ncolumns, figsize=(15, 10))

# Flattening the axes array for easy iteration
axes = axes.flatten()

# Iterating over each feature column and create a distribution plot in the corresponding axis
for a, column in enumerate(Data.columns[1:]):
    sns.histplot(data=Data, x=column, kde=True, ax=axes[a], color='lightblue')
    axes[a].set_title(column)
    axes[a].set_xlabel(column)
    axes[a].set_ylabel("Frequency")

# Hide any unused subplots
if len(Data.columns) < len(axes):
    for q in range(len(Data.columns), len(axes)):
        axes[q].axis("off")

# Adjusting the layout spacing
plt.tight_layout()

# Displaying the plots
plt.show()


# #### Interpretation:

# The data is normally distributed for most of the columns except for radius_se, parimeter_se, area_se, and compactness_se which are right skewed.

# #### Box plots for the Means Numerical Features by Group

# In[308]:


## Defining the rows and columns for the grid layout
rows = 2
columns = 5

# Create a grid layout of box plots for each feature grouped by "diagnosis"
fig, axes = plt.subplots(rows, columns, figsize=(15, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Features to be plotted
features_to_plot = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
                    "fractal_dimension_mean"]

# Iterate over each feature column and create a box plot in the corresponding axis
for a, column in enumerate(features_to_plot):
    sns.boxplot(x="diagnosis", y=column, data=MyData, ax=axes[a], palette={"M": "red", "B": "green"})
    axes[a].set_title(column)
    axes[a].set_xlabel("Diagnosis")
    axes[a].set_ylabel(column)

# Hide any unused subplots
if len(features_to_plot) < len(axes):
    for b in range(len(features_to_plot), len(axes)):
        axes[b].axis("off")

# Adjusting the layout spacing
plt.tight_layout()

# Displaying the plots
plt.show()


# **Findings:**
# 
# - Radius of the Malignant tumours (Cancerous cells) is larger than the Benign (Noncancerous cells) this suggests that cancerous cells tend to grow and spread, resulting in tumors with a larger overall size. In contrast, benign cells typically exhibit a smaller radius, indicating a more confined and localized growth pattern
# 
# - Tissue average of the Malignant (Cancerous cells) is larger than the Benign (Noncancerous cells) this implies that cancerous cells tend to occupy a greater extent of the tissue, potentially indicating more aggressive and invasive behavior. Benign cells, on the other hand, occupy a relatively smaller area within the tissue.
# 
# - Perimeter thickness of the Malignant (Cancerous cells) is larger than the Benign (Noncancerous cells) this indicates that the boundary or edges of cancerous tumors are typically more irregular and spread out, while benign cells tend to have a smoother and more well-defined perimeter.
# 
# - The area occupied by Malignant (Cancerous cells) is larger than Benign (Noncancerous cells).
# 
# - Compactness_mean for Malignant (Cancerous cells) is higher which indicates that the points on the mass are more closely packed together as compared to the Benign (Noncancerous cells).
# 
# - Concavity mean for Malignant (Cancerous cells) is larger than the Benign (Noncancerous cells) which indicates a greater severity of concave regions within the mass of Malignant cells.
# 
# - Concave point mean is much higher for the Malignant (Cancerous cells) indicating that there are more concave regions as compared to Benign (Noncancerous cells).
# 
# - The outliers have been ignored as they do not make any difference in the prediction.

# #### CORRELATION ANALYSIS
# 

# In[309]:


Corr_Mat= MyData.corr()


# In[310]:


fig, ab = plt.subplots(figsize = (18, 14))
ab = sns.heatmap(Corr_Mat,
                annot = True,
                cbar = False)


# **Positive correlations:**
# 
# - A significant positive correlation with a coefficient of 0.997855 was found between the variables "radius_mean" and "perimeter_mean" this is because an increase in the radius of a tumor (radius_mean) would generally result in an increase in its perimeter (perimeter_mean). This direct relationship leads to a strong positive correlation.
# 
# - The variables "radius_mean" and "area_mean" displayed a strong positive correlation, with a coefficient of 0.987357. This is because "area_mean" and "perimeter_mean" both depend on the size and shape of the tumor. As the tumor's size increases, both its area and perimeter are likely to increase, leading to a strong positive correlation between the two variables.
# 
# - A robust positive correlation was observed between "radius_mean" and "radius_worst," with a coefficient of 0.969539 this is because both variables capture the overall size or extent of the tumor. so, it is expected that as the mean radius of a tumor increases, the worst (largest) radius measurement would also tend to increase.
# 
# - The variables "area_mean" and "perimeter_mean" showed a strong positive correlation, with a coefficient of 0.986507. This shows that when area of tumor increases the perimeter will also increase which is quite obvious.
# 
# - A substantial positive correlation was detected between "area_mean" and "area_worst," with a coefficient of 0.959213.
# 
# - A strong positive correlation of 0.883121 was found between "compactness_mean" and "concavity_mean". This tells us that tumors that exhibit higher compactness tend to have more concave regions, resulting in a positive correlation between these variables. This correlation could indicate that tumors with a higher degree of compactness are more likely to have a greater level of concavity.
# 
# **Negative correlation:**
# 
# - A moderate negative correlation, with a coefficient of -0.311631, was observed between "radius_mean" and "fractal_dimension_mean". A negative correlation between these variables could indicate that tumors with larger cell radii tend to have less irregular or more uniform shapes, resulting in a lower fractal dimension.

# In[311]:


#Dropping column since they are highly correlated with eachother

columns_to_drop = ['radius_mean', 'perimeter_mean', 'area_worst', 'radius_worst', 'concavity_mean']
MyData = MyData.drop(columns=columns_to_drop)


# ## METHODS

# The data problem in this study was to accurately classify breast cancer cases based on a dataset containing various tumor characteristics. The goal was to develop a predictive model that could effectively distinguish between benign and malignant tumors.
# 
# To analyze breast cancer classification, this study employed several machine learning algorithms, including Random Forest, Decision Tree, Logistic Regression, KNeighbors Classifier, SVM, and Naive Bayes. These algorithms were selected due to their effectiveness in binary classification tasks. The dataset was preprocessed by label encoding the target variable, "diagnosis," and performing feature scaling using the StandardScaler function to ensure compatibility across the algorithms. The dataset was then split into training and testing sets with a 70:30 ratio. Each algorithm was trained on the training set and evaluated on the testing set using accuracy, precision, and recall metrics. Based on the evaluation results, the Logistic Regression algorithm demonstrated the highest performance in terms of accuracy and recall, making it the algorithm of choice for further analysis. The feature importance of the logistic regression model was assessed by examining the absolute coefficients, with higher magnitudes indicating stronger influences on the breast cancer prediction. Additionally, the performance of the selected model was visualized using a confusion matrix to gain insights into its predictive performance.

# #### Label Encoding for Target Variable

# In[312]:


from sklearn.preprocessing import LabelEncoder
# Create an instance of the LabelEncoder
LE = LabelEncoder()

# Use the LabelEncoder to transform the 'diagnosis' column of the DataFrame
MyData['diagnosis'] = LE.fit_transform(MyData.diagnosis)

# Display the head of the modified DataFrame
MyData.head()


# #### Splitting the data into target and predictor variables
# 

# In[313]:


predictors = MyData.drop(['diagnosis'], axis=1)
target = MyData.diagnosis


# #### Split the data into training and testing set

# In[314]:


from sklearn.model_selection import train_test_split as split
X_train, X_test, Y_train, Y_test = split(predictors, target, test_size=0.3, random_state=0)


# #### Feature Scaling for numerical variables

# In[315]:


from sklearn.preprocessing import StandardScaler
# Create an instance of the StandardScaler
sc = StandardScaler()

# Fit the scaler on the training data and transform it
X_train = sc.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test = sc.transform(X_test)


# ### MODELLING

# ### Applying Supervised Learning Algorithms

# In[316]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "KNeighbors Classifier": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# Create lists to store the results
classifier_list = []
accuracy_list = []
precision_list = []
recall_list = []

for clf_name, clf in classifiers.items():
    # Fitting on training set
    clf.fit(X_train, Y_train)

    # predictions on the training set
    Y_train_prediction = clf.predict(X_train)

    # Calculating  accuracy, precision, and recall on the training data
    accuracy_train = accuracy_score(Y_train, Y_train_prediction)
    precision_train = precision_score(Y_train, Y_train_prediction)
    recall_train = recall_score(Y_train, Y_train_prediction)

    # for predictions on the test set
    Y_test_prediction = clf.predict(X_test)

    # Calculating the accuracy, precision, and recall on the test set
    accuracy_test = accuracy_score(Y_test, Y_test_prediction)
    precision_test = precision_score(Y_test, Y_test_prediction)
    recall_test = recall_score(Y_test, Y_test_prediction)

    # Add the results to the lists
    classifier_list.append(clf_name)
    accuracy_list.append(accuracy_test)
    precision_list.append(precision_test)
    recall_list.append(recall_test)

# Create a DataFrame from the lists
data = {
    "Classifier": classifier_list,
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list
}
my_result = pd.DataFrame(data)

# Sorting by Accuracy
my_result = df.sort_values(by="Accuracy", ascending=False)

# Printing the sorted DataFrame
print(my_result)


# **Comparison Result:**
# 
# Since Recall is the ratio of true positive predictions to the total actual positives. It represents the model's ability to correctly identify all positive cases in the dataset. In breast cancer prediction, recall indicate the proportion of correctly identified malignant cases out of all actual malignant cases which is important because the cost of false negatives (incorrectly predicting in our case as benign) is high. Since Logistic Regression and SVM both have same highest Recall and accuracy, we will select one of them for further exploration. Let us select Logistic Regression Model.
# 

# ### LOGISTIC REGRESSION MODEL
# 

# #### List of important features

# In[317]:


import matplotlib.pyplot as plt

# Creating an instance of the LogisticRegression model
logisticreg = LogisticRegression()

# Fitting  model on train set
logisticreg.fit(X_train, Y_train)

# Making predictions on the train set
Y_train_prediction = logisticreg.predict(X_train)

# Calculating the accuracy on the training set
accuracy_train = accuracy_score(Y_train, Y_train_prediction)

# Making predictions on  test set
Y_train_prediction = logisticreg.predict(X_test)

# Getting feature importance from logistic regression model
feature_importance = logisticreg.coef_[0]

# Sorting the features and coefficients in descending order of absolute coefficients
sort_features = sorted(zip(X.columns, feature_importance), key=lambda x: abs(x[1]), reverse=True)

# Printing the important features with coefficients
print("Important Features:")
for labels, coef in sort_features:
    print(f"{labels}: {coef}")



# **Findings:**
# 
# The coefficients of the logistic regression model reveal the impact of each feature on the prediction of breast cancer classification. These coefficients represent the estimated change in the log-odds of having breast cancer for a one-unit increase in the corresponding feature, while holding other features constant.
# 
# By examining the list of coefficients, we can draw certain conclusions. Positive coefficients (such as radius_se, perimeter_worst, and concave points_worst) indicate that an increase in these features is associated with a higher likelihood of having breast cancer. On the other hand, negative coefficients (such as texture_mean, smoothness_mean, and symmetry_se) suggest that an increase in these features is linked to a lower likelihood of having breast cancer. The magnitude of the coefficients is also important. Larger magnitude coefficients imply a stronger influence on the prediction of breast cancer.
# 
# Since, radius_se has the highest magnitude and is positive so it is the most important feature for the prediction of breast cancer when using Logistic regression. It is a feature that quantifies the standard error of the mean of distances from the center to points on the perimeter of a breast mass.

# #### Plotting Confusion Matrix

# In[318]:


# Generate the confusion matrix
confusion_mat = confusion_matrix(Y_test, Y_test_pred)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")

# Setting labels
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# Displaying the plot
plt.show()


# - The top-left cell (106) represents the number of true negatives (TN). It indicates that there are 106 instances that were correctly predicted as the negative class.
# 
# - The top-right cell (2) represents the number of false positives (FP). It indicates that there are 2 instances that were incorrectly predicted as the positive class when they were actually negative.
# 
# - The bottom-left cell (3) represents the number of false negatives (FN). It indicates that there are 3 instances that were incorrectly predicted as the negative class when they were actually positive.
# 
# - The bottom-right cell (60) represents the number of true positives (TP). It indicates that there are 60 instances that were correctly predicted as the positive class.

# ## RESULTS

# The study evaluated the performance of several machine learning algorithms for the classification of breast cancer cases based on tumor characteristics. Among the models tested, Logistic Regression emerged as the top performer, exhibiting the highest accuracy and recall scores on both the training and test datasets. The Logistic Regression model achieved an impressive accuracy rate of 97%, indicating its capability to accurately predict breast cancer cases. Moreover, the recall score, which measures the model's ability to correctly identify positive cases, was also high at 95.2%, underscoring the model's effectiveness in detecting malignant tumors.
# 
# Additionally, the feature importance analysis revealed that the "radius_se" feature exhibited the highest magnitude and positive coefficient in the logistic regression model. This finding suggests that the variability in tumor radius size played a significant role in predicting breast cancer cases. Other influential features such as "smoothness_se," "texture_se," and "symmetry_se" also contributed to the overall predictive power of the model.
# 
# To provide a comprehensive overview of the logistic regression model's performance, a confusion matrix was employed for visualization. The confusion matrix displayed a relatively low number of false positives and false negatives, further supporting the model's accuracy and reliability.
# 
# In summary, the results underscore the effectiveness of the Logistic Regression algorithm in accurately classifying breast cancer cases based on tumor characteristics. These findings indicate the potential of the selected model as a valuable tool to aid medical professionals in the early detection and diagnosis of breast cancer, ultimately leading to improved patient outcomes and timely interventions.
# 

# ## CONCLUSION

# The Projecrt aimed to develop an accurate and efficient breast cancer prediction model using supervised learning algorithms. The dataset consisted of 569 instances with various features related to breast cancer diagnosis. The dataset was preprocessed by label encoding the target variable and performing feature scaling.
# 
# Descriptive analysis of the dataset revealed interesting findings, such as significant variations in the size, area, texture, concavity, and fractal dimension of the breast masses. These variations provided valuable insights into the characteristics of malignant and benign tumors, emphasizing the importance of considering multiple features when diagnosing breast cancer.
# 
# Through a comparative analysis of different algorithms, including logistic regression, support vector machines, decision trees, random forests, SVM, and Naive Bayes, the logistic regression algorithm demonstrated the highest performance in terms of accuracy and recall. Hence, it was chosen as the algorithm of choice for further analysis.
# 
# The logistic regression model highlighted the importance of the "radius_se" feature, which represents the standard error of the mean of distances from the center to points on the perimeter. This feature had the highest magnitude coefficient and was identified as the most influential feature for breast cancer prediction using logistic regression.
# 
# Evaluation of the logistic regression model using a confusion matrix showed good performance in correctly predicting both malignant and benign cases. The model achieved a high number of true negatives and true positives, indicating its ability to accurately classify both types of cases. However, there were a small number of false positives and false negatives, suggesting room for further improvement in future iterations.

# <div align="center"> The End
