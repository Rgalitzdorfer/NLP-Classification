#Import Libraries
import os #Operating System
import pandas as pd #Data Manipulation
from sklearn.preprocessing import StandardScaler, OneHotEncoder #Encoding Categorical Variables
from sklearn.compose import ColumnTransformer #Preprocessing Pipeline
from sklearn.pipeline import Pipeline #Machine Learning Pipeline
from sklearn.model_selection import StratifiedKFold #Cross-Validation
from sklearn.ensemble import RandomForestClassifier #Random Forest Classification
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score #Evaluation Metrics
from sklearn.model_selection import GridSearchCV #Hyperparameter Tuning
import numpy as np #Arrays
import warnings #Suppressing Warnings
from imblearn.over_sampling import BorderlineSMOTE #Imbalanced Datasets
from imblearn.pipeline import Pipeline as ImbPipeline #Imbalanced Learning Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer #Text Feature Extraction

#Warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only')

#Directories
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv' 
output_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Weeks 7-8' 
os.makedirs(output_directory, exist_ok=True) 

data = pd.read_csv(file_path) #Read CSV
print("First few rows of the dataset:") #Error Detection
print(data.head()) 
print("\nMissing values in each column:") #Error Detection
print(data.isnull().sum()) 
data = data.dropna(subset=['State']) #Drop Rows with Missing Information
data['Text'] = data['Text'].fillna('') #Fill Missing Text Values
print("\nDataset information after cleaning:") 
print(data.info()) #Statistics

#Define Text
text_feature_column = 'Text' 
categorical_features = ['Rule_type', 'Rule_order'] #Categorical Features
numerical_features = ['Updated_True_Time', 'Correct', 'First.Action', 'Attempt.Count', 'NormalizedFirstRT'] #Numerical Features

#TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer() 
text_transformer = Pipeline(steps=[
    ('tfidf', tfidf_vectorizer) 
])

#Machine Learning
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), #Scale Numerical Features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), #Encode Categorical Features
        ('txt', text_transformer, text_feature_column) #Transform Text Features
    ],
    remainder='drop' 
)
X = data.drop(columns=['State']) #Feature Set
y = data['State'] #Target Variable
X_tfidf = tfidf_vectorizer.fit_transform(X[text_feature_column]) #Fit TF-IDF Vectorizer
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=2) #Initialize BorderlineSMOTE
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced_subsample') #Initialize Random Forest
#Define Parameter Grid
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10] 
}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Initialize Cross-Validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2) #Initialize Grid Search
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor), 
    ('borderline_smote', borderline_smote), 
    ('classifier', grid_search) 
])
pipeline.fit(X, y) #Fit Pipeline
best_params = pipeline.named_steps['classifier'].best_params_ #Get Best Parameters
print(f"\nBest parameters: {best_params}") 
rf_classifier_optimized = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced_subsample') #Initialize Optimized Random Forest
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Initialize Cross-Validation
#Initialize
all_predictions = [] 
accuracy_scores = [] 
balanced_accuracy_scores = [] 
precision_scores = [] 
f1_scores = [] 
feature_importances = [] 

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] #Split Data
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] #Split Target
    X_train_transformed = preprocessor.fit_transform(X_train) #Transform Training Data
    X_test_transformed = preprocessor.transform(X_test) #Transform Test Data
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_transformed, y_train) #Oversample Training Data
    rf_classifier_optimized.fit(X_train_resampled, y_train_resampled) #Fit Model
    y_pred = rf_classifier_optimized.predict(X_test_transformed) #Predict Test Data
    #Create DataFrame
    fold_predictions_df = pd.DataFrame({
        'Actual': y_test, #Actual Values
        'Predicted': y_pred #Predicted Values
    })
    all_predictions.append(fold_predictions_df) 
    #Evaluation Metrics
    acc = accuracy_score(y_test, y_pred) 
    bal_acc = balanced_accuracy_score(y_test, y_pred) 
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) 
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) 
    #Append
    accuracy_scores.append(acc) 
    balanced_accuracy_scores.append(bal_acc) 
    precision_scores.append(prec) 
    f1_scores.append(f1) 
    feature_importances.append(rf_classifier_optimized.feature_importances_) 
    print(f"\nFold {fold+1} evaluation:") 
    print(f"Accuracy: {acc}") 
    print(f"Balanced Accuracy: {bal_acc}") 
    print(f"Precision: {prec}") 
    print(f"F1 Score: {f1}") 
    print(classification_report(y_test, y_pred, zero_division=0)) 

#Create DataFrame
combined_predictions = pd.concat(all_predictions, axis=0) 
combined_predictions.to_csv(f'{output_directory}/Predictions_RF.csv', index=False) #Save CSV
print("\nAverage Metrics Across All Folds:") 
print(f"Average Accuracy: {np.mean(accuracy_scores)}") 
print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_scores)}") 
print(f"Average Precision: {np.mean(precision_scores)}") 
print(f"Average F1 Score: {np.mean(f1_scores)}") 

max_length = max(len(f) for f in feature_importances) #Get Max Array Length
padded_importances = [np.pad(f, (0, max_length - len(f)), 'constant') for f in feature_importances] 
avg_feature_importances = np.mean(np.vstack(padded_importances), axis=0) #Average Feature Importances
print("\nAverage Feature Importances:") 
print(avg_feature_importances) 
tfidf_length = text_transformer.named_steps['tfidf'].vocabulary_.__len__() #Get TF-IDF Length
tfidf_feature_importance = avg_feature_importances[:tfidf_length] #Get TF-IDF Importances
other_feature_importance = avg_feature_importances[-len(numerical_features + categorical_features):] 
print(f"\nTF-IDF Feature Importance: {np.mean(tfidf_feature_importance)}") 
print(f"Other Features Importance: {np.mean(other_feature_importance)}") 
