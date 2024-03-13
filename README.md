# Breast Cancer Detection
![image](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Prediction/assets/139073097/c826f624-1c0c-4f91-84b9-342074b255c5)
### Project Overview 💡
This project focuses on leveraging supervised learning models to develop an accurate breast cancer prediction system. By analyzing different features extracted from breast cancer patients, including known cases labeled as Malignant or Benign, the aim is to assess and compare various algorithms such as logistic regression, support vector machines, decision trees, random forests, SVM, and Naive Bayes. Evaluation metrics like accuracy, precision, and recall will guide the assessment process. The results of this study hold significant promise in advancing breast cancer diagnosis and improving medical decision-making, emphasizing the importance of early detection and precise diagnosis for enhanced treatment outcomes.
### Data Sources 📊
The dataset is obtained from [kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) and is available in this repository [dataset](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/tree/main/Dataset) as well.
### Tools Used 🧰
- Numpy [Info](https://numpy.org/)
- Pandas [info](https://pandas.pydata.org/)
- Matplotlib [info](https://matplotlib.org/)
- seaborn [info](https://seaborn.pydata.org/)
- sklearn.preprocessing [info](https://scikit-learn.org/stable/)
### Dataset Explained 
The Kaggle breast cancer dataset consists of 569 rows and 32 columns, each representing features relevant to diagnosing breast cancer. Each entry is identified by a unique "id" and accompanied by a "diagnosis" label indicating whether the breast mass is Malignant (cancerous) or Benign (non-cancerous). Features like "radius_mean," "texture_mean," "perimeter_mean," and "area_mean" provide insights into the size, texture, and extent of the mass. Other features include measures of smoothness, compactness, concavity, and concave points, with variations for "worst" and "standard error" values. Additional features such as symmetry and fractal dimension enrich the dataset, offering insights into the characteristics of breast masses. The "diagnosis" column has two unique values, while other columns exhibit varying numbers of unique values, reflecting the diversity of measurements extracted from breast cancer data.
### Data Cleaning/Preparation 🧹
In the data cleaning and preparation phase, we verified that the dataset was in suitable condition for analysis. Most variables retained their original data types, with the exception of the Diagnosis target variable, which had to be transformed into binary values (1 for Malignant and 0 for Benign) to facilitate classification tasks later in the project. We had ensured that the dataset contained no null values, eliminating potential sources of error or bias in the analysis. Additionally, we had confirmed the absence of duplicate entries, ensuring the integrity and reliability of the dataset for subsequent stages of analysis. These meticulous steps laid a solid foundation for our exploration of various supervised learning algorithms and the development of an accurate breast cancer prediction model.
### Exploratory Data Analysis
- The dataset exhibits significant variation in the mean radius of breast masses, ranging from 6.98 to 28.11 units, indicating diverse sizes among instances.
- The mean area of breast masses spans a wide range from 143.50 to 2501.00 square units, showcasing substantial diversity in size and extent.
- The average smoothness of breast masses varies between 0.0526 and 0.1634, suggesting a broad spectrum of surface textures from smoother to rougher.
- Breast masses also show variability in mean concavity, with values ranging from 0 to 0.4268, indicating differences in shape complexity.
- The mean fractal dimension varies between 0.04996 and 0.09744, reflecting differences in geometric patterns and intricacy.
These findings underscore the importance of considering multiple features when analyzing and predicting breast cancer, as variations in size, area, texture, concavity, and fractal dimension provide valuable insights for diagnosis and further exploration.

![Distribution of Diagnosis](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/assets/139073097/cf54da64-ac6a-413f-a2c3-d368055d25cd)

There were 62.7% Benign (Noncancerous) cells and 37.3% Malignant (Cancerous) cells in the dataset.
#### Numerical Features
![Histogram for numerical features](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/assets/139073097/fd7fc82b-09f5-44bf-8ce0-16ea4ae2ec70)

The data is normally distributed for most of the columns except for radius_se,
parimeter_se, area_se, and compactness_se which are right skewed.
#### Box plots for the Means Numerical Features by Group
![Box plots for the Means Numerical Features by Group](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/assets/139073097/e14624e7-9b97-4b90-b303-9a81692b67f3)

##### Findings:
- Radius of the Malignant tumours (Cancerous cells) is larger than the Benign (Noncancerous
cells) this suggests that cancerous cells tend to grow and spread, resulting in tumors with a
larger overall size. In contrast, benign cells typically exhibit a smaller radius, indicating a
more confined and localized growth pattern
- Tissue average of the Malignant (Cancerous cells) is larger than the Benign (Noncancerous
cells) this implies that cancerous cells tend to occupy a greater extent of the tissue, potentially
indicating more aggressive and invasive behavior. Benign cells, on the other hand, occupy a
relatively smaller area within the tissue.
- Perimeter thickness of the Malignant (Cancerous cells) is larger than the Benign (Noncancerous
cells) this indicates that the boundary or edges of cancerous tumors are typically more
irregular and spread out, while benign cells tend to have a smoother and more well-defined
perimeter.
- The area occupied by Malignant (Cancerous cells) is larger than Benign (Noncancerous cells).
- Compactness_mean for Malignant (Cancerous cells) is higher which indicates that the points
on the mass are more closely packed together as compared to the Benign (Noncancerous cells).
- Concavity mean for Malignant (Cancerous cells) is larger than the Benign (Noncancerous
cells) which indicates a greater severity of concave regions within the mass of Malignant cells.
- Concave point mean is much higher for the Malignant (Cancerous cells) indicating that there
are more concave regions as compared to Benign (Noncancerous cells).
- The outliers have been ignored as they do not make any difference in the prediction

#### Correlation Analysis
![CORRELATION ANALYSIS](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/assets/139073097/d8ac7489-1061-4df6-a109-7ca69321fd61)

##### Positive correlations:
- A significant positive correlation with a coefficient of 0.997855 was found between the variables
“radius_mean” and “perimeter_mean” this is because an increase in the radius of a tumor
(radius_mean) would generally result in an increase in its perimeter (perimeter_mean). This
direct relationship leads to a strong positive correlation.
- The variables “radius_mean” and “area_mean” displayed a strong positive correlation, with
a coefficient of 0.987357. This is because “area_mean” and “perimeter_mean” both depend
on the size and shape of the tumor. As the tumor’s size increases, both its area and perimeter
are likely to increase, leading to a strong positive correlation between the two variables.
- A robust positive correlation was observed between “radius_mean” and “radius_worst,” with
a coefficient of 0.969539 this is because both variables capture the overall size or extent of the
tumor. so, it is expected that as the mean radius of a tumor increases, the worst (largest)
radius measurement would also tend to increase.
- The variables “area_mean” and “perimeter_mean” showed a strong positive correlation, with
a coefficient of 0.986507. This shows that when area of tumor increases the perimeter will
also increase which is quite obvious.
- A substantial positive correlation was detected between “area_mean” and “area_worst,” with
14
a coefficient of 0.959213.
- A strong positive correlation of 0.883121 was found between “compactness_mean” and “concavity_
mean”. This tells us that tumors that exhibit higher compactness tend to have more
concave regions, resulting in a positive correlation between these variables. This correlation
could indicate that tumors with a higher degree of compactness are more likely to have a
greater level of concavity.
##### Negative correlation:
- A moderate negative correlation, with a coefficient of -0.311631, was observed between “radius_
mean” and “fractal_dimension_mean”. A negative correlation between these variables
could indicate that tumors with larger cell radii tend to have less irregular or more uniform
shapes, resulting in a lower fractal dimension.
#### Methods Used
The data problem in this study was to accurately classify breast cancer cases based on a dataset
containing various tumor characteristics. The goal was to develop a predictive model that could
effectively distinguish between benign and malignant tumors.
To analyze breast cancer classification, this study employed several machine learning algorithms,
including Random Forest, Decision Tree, Logistic Regression, KNeighbors Classifier, SVM, and
Naive Bayes. These algorithms were selected due to their effectiveness in binary classification tasks.
The dataset was preprocessed by label encoding the target variable, “diagnosis,” and performing
feature scaling using the StandardScaler function to ensure compatibility across the algorithms.
The dataset was then split into training and testing sets with a 70:30 ratio. Each algorithm was
trained on the training set and evaluated on the testing set using accuracy, precision, and recall
metrics.
### Modelling Result
Under the Modelling heading, the study implemented various supervised learning algorithms to classify breast cancer cases based on tumor characteristics.Each algorithm was trained and evaluated using accuracy, precision, and recall metrics. Logistic Regression and SVM exhibited the highest recall and accuracy scores, indicating their ability to correctly identify malignant cases, which is crucial in breast cancer prediction due to the high cost of false negatives. Given their comparable performance, either Logistic Regression or SVM was selected for further exploration.

#### Logistic Regression
Based on the evaluation results, the Logistic Regression algorithm demonstrated the
highest performance in terms of accuracy and recall, making it the algorithm of choice for further
analysis. The feature importance of the logistic regression model was assessed by examining the
absolute coefficients, with higher magnitudes indicating stronger influences on the breast cancer
prediction.
#### Feature Importance
By employing logistic regression, the coefficients associated with each feature were determined, showcasing their impact on the likelihood of breast cancer presence. Positive coefficients, like those for radius_se, perimeter_worst, and concave points_worst, suggest an increase in these features correlates with a higher likelihood of breast cancer. Conversely, negative coefficients, such as texture_mean, smoothness_mean, and symmetry_se, indicate an increase in these features corresponds to a lower likelihood of breast cancer. The magnitude of the coefficients also plays a crucial role, with larger magnitudes indicating stronger influences on the prediction. Among these features, radius_se emerges as the most influential, given its highest magnitude and positive coefficient. Radius_se quantifies the standard error of the mean of distances from the center to points on the perimeter of a breast mass, thus serving as a significant predictor for breast cancer classification in logistic regression models.

##### Confusion Matrix
![Confusion Matrix](https://github.com/MuhammadAhsanBughio/Breast-Cancer-Detection/assets/139073097/b984bdc9-58f2-4072-a06f-e4c77d5f225b)

- The top-left cell (106) represents the number of true negatives (TN). It indicates that there
are 106 instances that were correctly predicted as the negative class.
- The top-right cell (2) represents the number of false positives (FP). It indicates that there
are 2 instances that were incorrectly predicted as the positive class when they were actually
negative.
- The bottom-left cell (3) represents the number of false negatives (FN). It indicates that there
are 3 instances that were incorrectly predicted as the negative class when they were actually
positive.
- The bottom-right cell (60) represents the number of true positives (TP). It indicates that
there are 60 instances that were correctly predicted as the positive class.

### Result/Conclusion
The study evaluated multiple machine learning algorithms for breast cancer classification, with Logistic Regression emerging as the top performer, achieving 97% accuracy and 95.2% recall. The "radius_se" feature was identified as most influential, indicating its significant role in prediction. A confusion matrix illustrated the model's reliability with minimal false positives and negatives. Logistic Regression proved effective for early detection and diagnosis, holding promise for improved patient outcomes. The project aimed to develop an accurate breast cancer prediction model, highlighting variations in tumor characteristics and emphasizing the importance of considering multiple features. Logistic Regression's success underscores its potential as a valuable diagnostic tool, though further enhancements are possible.
