# Import Libraries
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

# Warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only')

# Directories
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv'
output_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 7'
os.makedirs(output_directory, exist_ok=True)

data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(data.head())
print("\nMissing values in each column:")
print(data.isnull().sum())
data = data.dropna(subset=['State'])
data['Text'] = data['Text'].fillna('')
print("\nDataset information after cleaning:")
print(data.info())

# Define Text
text_feature_column = 'Text'
categorical_features = ['Rule_type', 'Rule_order']
numerical_features = ['Updated_True_Time', 'Correct', 'First.Action', 'Attempt.Count', 'NormalizedFirstRT']

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
text_transformer = Pipeline(steps=[
    ('tfidf', tfidf_vectorizer)
])

# Machine Learning
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('txt', text_transformer, text_feature_column)
    ],
    remainder='drop'
)

X = data.drop(columns=['State'])
y = data['State']
X_tfidf = tfidf_vectorizer.fit_transform(X[text_feature_column])
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=2)
logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)  # Initialize Logistic Regression

# Define Parameter Grid
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('borderline_smote', borderline_smote),
    ('classifier', grid_search)
])

pipeline.fit(X, y)
best_params = pipeline.named_steps['classifier'].best_params_
print(f"\nBest parameters: {best_params}")

logistic_regression_optimized = LogisticRegression(**best_params, multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize
all_predictions = []
accuracy_scores = []
balanced_accuracy_scores = []
precision_scores = []
f1_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_transformed, y_train)
    logistic_regression_optimized.fit(X_train_resampled, y_train_resampled)
    y_pred = logistic_regression_optimized.predict(X_test_transformed)

    fold_predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    all_predictions.append(fold_predictions_df)

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    accuracy_scores.append(acc)
    balanced_accuracy_scores.append(bal_acc)
    precision_scores.append(prec)
    f1_scores.append(f1)

    print(f"\nFold {fold+1} evaluation:")
    print(f"Accuracy: {acc}")
    print(f"Balanced Accuracy: {bal_acc}")
    print(f"Precision: {prec}")
    print(f"F1 Score: {f1}")
    print(classification_report(y_test, y_pred, zero_division=0))

combined_predictions = pd.concat(all_predictions, axis=0)
combined_predictions.to_csv(f'{output_directory}/Predictions_LogReg.csv', index=False)
print("\nAverage metrics across all folds:")
print(f"Average Accuracy: {np.mean(accuracy_scores)}")
print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_scores)}")
print(f"Average Precision: {np.mean(precision_scores)}")
print(f"Average F1 Score: {np.mean(f1_scores)}")

# Print a correlation matrix
# Select only the numerical features for correlation matrix
numerical_data = data[numerical_features].copy()

# Add any additional numeric columns if they exist
for column in data.columns:
    if column not in numerical_features and pd.api.types.is_numeric_dtype(data[column]):
        numerical_data[column] = data[column]

correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(f'{output_directory}/Correlation_Matrix_LogReg.png')  # Save the plot
plt.show()
