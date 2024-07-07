#Import Libraries
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

#Warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only')

#Directories
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv' 
output_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 7'  
os.makedirs(output_directory, exist_ok=True)  

#Read & Process Data
data = pd.read_csv(file_path)  
print(data.head())  
data = data.dropna(subset=['State']) 
data['Text'] = data['Text'].fillna('') 
print("\nDataset Information After Cleaning:")
print(data.info())  

#Define Features
text_feature_column = 'Text'  
categorical_features = ['Rule_type', 'Rule_order']  
numerical_features = ['Updated_True_Time', 'Correct', 'First.Action', 'Attempt.Count', 'NormalizedFirstRT']  

#TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer() #Initialize 
text_transformer = Pipeline(steps=[
    ('tfidf', tfidf_vectorizer)
]) #Pipeline for Text Transformation

#Machine Learning
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), #Standardize Numerical Features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), #One-Hot Encode Categorical Features
        ('txt', text_transformer, text_feature_column) #Apply TF-IDF to Text Feature
    ],
    remainder='drop'
)  

X = data.drop(columns=['State']) #Drop Target Column
y = data['State'] #Define Target Vector
X_tfidf = tfidf_vectorizer.fit_transform(X[text_feature_column]) #Fit & Transform Text Data Using TF-IDF
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=2) #Initialize Borderline SMOTE for Class Imbalance
logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000) #Initialize Logistic Regression

#Define Parameter Grid
param_grid = {
    'C': [0.1, 1, 10],  
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Cross-Validation 
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2) #Initialize GridSearchCV
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor), #Add Preprocessor to Pipeline
    ('borderline_smote', borderline_smote), #Add Borderline SMOTE to Pipeline
    ('classifier', grid_search) #Add GridSearchCV to Pipeline
])

pipeline.fit(X, y) #Fit Pipeline 
best_params = pipeline.named_steps['classifier'].best_params_  #Extract Best Parameters 
print(f"\nBest Parameters: {best_params}")
logistic_regression_optimized = LogisticRegression(**best_params, multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42) #Optimized Logistic Regression
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Define Cross-Validation Strategy
#Initialize
all_predictions = []
accuracy_scores = []
balanced_accuracy_scores = []
precision_scores = []
f1_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)): #Perform Cross-Validation
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy() #Split Features into Train & Test
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy() #Split Target into Train & Test 
    X_train_transformed = preprocessor.fit_transform(X_train) #Fit & Transform Training Data
    X_test_transformed = preprocessor.transform(X_test) #ransform Testing Data
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_transformed, y_train) #Apply Borderline SMOTE 
    logistic_regression_optimized.fit(X_train_resampled, y_train_resampled) #Fit Optimized Logistic Regression Model
    y_pred = logistic_regression_optimized.predict(X_test_transformed) #Predict Target 

    fold_predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })  # Create a DataFrame for Fold Predictions
    all_predictions.append(fold_predictions_df) #Append Fold Predictions

    #Evaluation Metrics
    acc = accuracy_score(y_test, y_pred) #Calculate Accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred) #Calculate Balanced Accuracy
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) #Calculate Precision
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) #Calculate F1 Score
    accuracy_scores.append(acc) #Append Accuracy Score to List
    balanced_accuracy_scores.append(bal_acc) #Append Balanced Accuracy Score to List
    precision_scores.append(prec) #Append Precision Score to List
    f1_scores.append(f1) #Append F1 Score to List
    #Print Statements
    print(f"\nFold {fold+1} Evaluation:")
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")
    print(f"Precision: {prec}")
    print(f"F1 Score: {f1}")
    print(classification_report(y_test, y_pred, zero_division=0)) #Print Classification Report

combined_predictions = pd.concat(all_predictions, axis=0) #Concatenate All Predictions
combined_predictions.to_csv(f'{output_directory}/Predictions_LogReg.csv', index=False) #Save to CSV
print("\nAverage Metrics Across All Folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores)}")
print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_scores)}")
print(f"Average Precision: {np.mean(precision_scores)}")
print(f"Average F1 Score: {np.mean(f1_scores)}")

#Correlation Matrix
numerical_data = data[numerical_features].copy()
for column in data.columns:
    if column not in numerical_features and pd.api.types.is_numeric_dtype(data[column]):
        numerical_data[column] = data[column]

correlation_matrix = numerical_data.corr() #Calculate Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  
plt.title('Correlation Matrix')
plt.savefig(f'{output_directory}/Matrix_LogReg.png') 
plt.show()
