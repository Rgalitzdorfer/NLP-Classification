#NLP Classification Overview
In conjunction with FACETLab (Future Adaptive Collaborative Educational Technologies), this project aims to predict learning outcomes from behavioral and log data received from participants performing various cognitive tests. Additionally, audio recordings from each participant were kept to track what was being said during different parts of the examination. This 'textual' data was merged using an algorithm that most closely aligns the textual data with the timestamps from both the behavioral and log data. Then, TF-IDF Vectorizer was used to convert text into vectors based on the relevance of particular words. Lastly, this combined dataset used Multiclass Logistic Regression to predict and classify each participant's cognitive state at specific times in the experiment. 

##Problems:
1. Data 



##Results:
The Multiclass Logistic Regression model achieved a 95% accuracy score, 80% precision score, and 77% F1 score. K-fold cross-validation was used with 10 splits to ensure consistent results across all different folds. Since 1 cognitive state occurred more frequently than the other 3, Borderline SMOTE was used as an oversampling technique to help improve the accuracy of predicting less frequent states. 

#Code Breakdown
##Week 1: Exploratory Data Analysis
