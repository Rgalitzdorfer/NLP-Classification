# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only')

# File path
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/Feature_Matrix.csv'

# Step 1: Data Exploration and Cleaning
# Load the dataset
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing 'State' values, as this is our target variable
data = data.dropna(subset=['State'])

# Display the updated summary
print("\nDataset information after cleaning:")
print(data.info())

# Step 2: Feature Engineering
# Identify text feature columns by checking columns that have words as column names
text_feature_columns = data.columns[10:]  # Assuming the first 10 columns are not text features

# Create a column transformer for handling categorical and numerical features
categorical_features = ['Rule_type', 'Rule_order']
numerical_features = ['Updated_True_Time']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('txt', StandardScaler(), text_feature_columns)
    ],
    remainder='drop'
)

# Prepare the features and target variable
X = data.drop(columns=['State'])
y = data['State']

# Transform the features
X_transformed = preprocessor.fit_transform(X)

# Step 3: Model Building and Cross-Validation
# Define the parameter grid for GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Create a GridSearchCV object with StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)

# Fit the grid search to the data
print("\nStarting grid search...")
grid_search.fit(X_transformed, y)

# Get the best parameters
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# Train the model with the best parameters
rf_classifier_optimized = RandomForestClassifier(**best_params, random_state=42)

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store results of each fold
all_predictions = []

for fold, (train_index, test_index) in enumerate(skf.split(X_transformed, y)):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf_classifier_optimized.fit(X_train, y_train)
    y_pred = rf_classifier_optimized.predict(X_test)
    
    # Save fold predictions
    fold_predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Optional: include some features in the output for context
    feature_columns = [f'Feature_{i}' for i in range(1, 6)]
    features_df = pd.DataFrame(X_test[:, :5], columns=feature_columns)
    predictions_with_features = pd.concat([features_df, fold_predictions_df.reset_index(drop=True)], axis=1)
    
    # Save detailed predictions to a single CSV file for this fold
    predictions_with_features.to_csv(f'/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/Detailed_Predictions_Fold_{fold+1}.csv', index=False)
    
    all_predictions.append(predictions_with_features)
    
    # Print evaluation metrics
    print(f"\nFold {fold+1} evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0)}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='macro', zero_division=0)}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Combine all fold predictions
combined_predictions = pd.concat(all_predictions, axis=0)

# Save combined detailed predictions to a single CSV file
combined_predictions.to_csv('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/Combined_Detailed_Predictions.csv', index=False)
