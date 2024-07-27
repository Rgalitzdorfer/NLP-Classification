# NLP Classification Overview
In conjunction with FACETLab (Future Adaptive Collaborative Educational Technologies), this project aims to predict learning outcomes from behavioral and log data received from participants performing various cognitive tests. Additionally, audio recordings from each participant were recorded to track what was being said during different parts of the examination. This 'textual' data was merged using an algorithm that most closely aligns the textual data with the timestamps from both the behavioral and log data. Then, TF-IDF Vectorizer was used to convert text into vectors based on the relevance of particular words. Lastly, this combined dataset used Multiclass Logistic Regression to predict and classify each participant's cognitive state at specific times throughout the experiment. 

## Largest Problems Faced:
1. Building an algorithm to distribute and allocate textual data using mismatched timestamps from the log data as closely as possible.
2. Selecting the best NLP approach for the textual data that would work best given the nature of the data and the scope of the presumptive classification models.
3. Accounting for less frequently occurring cognitive states with Borderline SMOTE to reduce the model's bias in predicting the majority class.

## Results Achieved:
The Multiclass Logistic Regression model achieved a 95% accuracy score, 79% balanced accuracy score, 80% precision score, and 77% F1 score. K-fold cross-validation was used with 10 splits to ensure consistent results across all different folds. Since 1 cognitive state occurred more frequently than the other 3, Borderline SMOTE was used as an oversampling technique to help improve the accuracy of predicting less frequent states. 


# Code Breakdown
## Libraries Used
### Pandas (DataFrame Manipulation)
### Numpy (Arrays)
### Matplotlib (Data Visualization)
### Seaborn (Data Visualization)
### Datetime (Matching Timestamps)
### TfidfVectorizer (Natural Language Processing)
### Borderline SMOTE (Class Imbalance)
### StratifiedKFold (Model Evaluation)
### Skicit-Learn (Machine Learning)

## Week 1: Exploratory Data Analysis
EDA was performed on various columns from each participant's log data to get an understanding of the data's distribution, the importance of different features, and the relationship between cognitive state and other factors. This was later used to determine which features would be most useful for the classification model.

## Week 2: Data Cleaning
This process consisted of merging 'Step 0' log data for each participant which will be used to accurately align the textual data timestamps with each Problem & Problem ID. Additionally, traditional modifications such as removing null values, sorting by identifier variables, and converting the data to its most useful format were also completed.

## Weeks 3-6: Feature Engineering
Throughout these weeks the main objective was to build an algorithm that assigned and allocated each participant's textual data with their log and behavioral data. This required several different steps such as pairing the text data with the closest timestamp of log data, then distributing it proportionally amongst all of the same rows of log data from the same Problem & Problem ID, and lastly mitigating data leakage through tracking total word counts for each participant throughout these steps. In week 6, behavioral data was added for each unique Problem & Problem ID. Several different Natural Language Processing techniques were experimented with to represent the textual data such as the Bag of Words approach, Word Emedding, and ultimately the TF-IDF Vectorizer which was most successful. This finished the preprocessing stage in preparation for model building.

## Week 7: Model Building
Three different classification models were tested on the complete dataset containing all merged and cleaned data from every participant. They were the Support Vector Machine Classifier, the Random Forest Classifier, and the Multinomial Logistic Regression Classifier. All models used Borderline SMOTE for oversampling of the minority classes and k-fold cross-validation to mitigate inaccurate results. The performance of each classification model is listed below:
### Random Forest: 
92% Overall Accuracy, 55% Balanced Accuracy, 49% Precision Score, 51% F1 Score
### Support Vector Machine: 
94% Overall Accuracy, 75% Balanced Accuracy, 76% Precision Score, 74% F1 Score
### Multiclass Logistic Regression (Top Performer):
95% Overall Accuracy, 79% Balanced Accuracy, 80% Precision Score, 77% F1 Score

## Week 8: Model Evaluation
To better understand what features were contributing to the success of the classification models, both the Support Vector Machine & Multiclass Logistic Regression models were performed without the NLP from the textual data to view the results without the TF-IDF Vectorizer. The model evaluation indicates that Multiclass Logistic Regression was more dependent on the textual features for the success of the model as the model without NLP performed 6-8% worse on all advanced evaluation metrics. The performance of each classification model without textual data is listed below:
### Support Vector Machine:
94% Overall Accuracy, 76% Balanced Accuracy, 79% Precision Score, 76% F1 Score
### Multiclass Logistic Regression: 
92% Overall Accuracy, 73% Balanced Accuracy, 72% Precision Score, 69% F1 Score
