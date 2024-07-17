#Import Libraries
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix
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
output_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Weeks 7-8'
os.makedirs(output_directory, exist_ok=True)

#Read CSV
data = pd.read_csv(file_path)
data = data.dropna(subset=['Label']) #Make sure 'Label' Column is Used
data['Text'] = data['Text'].fillna('')
print(data.info())

#Identify Cognitive Labels
unique_labels = data['Label'].unique()
print(f"Unique 'Label' values and their labels: {unique_labels}")

#Define Features
text_feature_column = 'Text'
categorical_features = ['Rule_type', 'Rule_order']
numerical_features = ['Updated_True_Time', 'Correct', 'First.Action', 'Attempt.Count', 'NormalizedFirstRT']

#TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
text_transformer = Pipeline(steps=[
    ('tfidf', tfidf_vectorizer)
])

#Machine Learning
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('txt', text_transformer, text_feature_column)
    ],
    remainder='drop'
)

X = data.drop(columns=['Label']) #Drop Target Column
y = data['Label'].map({label: idx for idx, label in enumerate(unique_labels)}) #Map labels to numerical values
X_tfidf = tfidf_vectorizer.fit_transform(X[text_feature_column]) #Fit & Transform
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=2) #Borderline SMOTE
svm_classifier = SVC(kernel='linear', probability=True) #Initialize SVM

#Parameter Grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Cross-Validation Strategy
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2) #Grid Search
pipeline = ImbPipeline(steps=[ #Pipeline
    ('preprocessor', preprocessor),
    ('borderline_smote', borderline_smote),
    ('classifier', grid_search)
])
pipeline.fit(X, y) #Fit Pipeline
best_params = pipeline.named_steps['classifier'].best_params_ #Extract Best Parameters
print(f"\nBest Parameters: {best_params}")
svm_classifier_optimized = SVC(**best_params, kernel='linear', probability=True, random_state=42) #Optimized SVM
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Cross Validation

#Initialize
all_predictions = []
accuracy_scores = []
balanced_accuracy_scores = []
precision_scores = []
f1_scores = []
confusion_matrices = []

#Iterate Through Each Fold
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_transformed, y_train)
    svm_classifier_optimized.fit(X_train_resampled, y_train_resampled)
    y_pred = svm_classifier_optimized.predict(X_test_transformed)
    fold_predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    all_predictions.append(fold_predictions_df)

    #Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy_scores.append(acc)
    balanced_accuracy_scores.append(bal_acc)
    precision_scores.append(prec)
    f1_scores.append(f1)

    #Print Results
    print(f"\nFold {fold+1} Evaluation:")
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")
    print(f"Precision: {prec}")
    print(f"F1 Score: {f1}")
    print(classification_report(y_test, y_pred, zero_division=0))

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

combined_predictions = pd.concat(all_predictions, axis=0) #Combine Results
combined_predictions.to_csv(f'{output_directory}/Predictions_SVM.csv', index=False) #Save to CSV
print("\nAverage Metrics Across All Folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores)}")
print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_scores)}")
print(f"Average Precision: {np.mean(precision_scores)}")
print(f"Average F1 Score: {np.mean(f1_scores)}")

#Average Confusion Matrices Over All Folds
max_classes = max(cm.shape[0] for cm in confusion_matrices)

#Ensure Fold Length is Correct
def pad_confusion_matrix(cm, max_classes):
    padded_cm = np.zeros((max_classes, max_classes))
    padded_cm[:cm.shape[0], :cm.shape[1]] = cm
    return padded_cm
confusion_matrices = [pad_confusion_matrix(cm, max_classes) for cm in confusion_matrices]
average_cm = np.mean(confusion_matrices, axis=0)

#Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(average_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{output_directory}/Confusion_Matrix_SVM.png')
plt.show()
