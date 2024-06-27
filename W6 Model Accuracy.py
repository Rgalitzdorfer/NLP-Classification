#Import Libraries
import os #Operating System
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For feature scaling and encoding categorical variables
from sklearn.compose import ColumnTransformer  # For combining different preprocessing pipelines
from sklearn.pipeline import Pipeline  # For creating a machine learning pipeline
from sklearn.model_selection import StratifiedKFold  # For cross-validation
from sklearn.ensemble import RandomForestClassifier  # For random forest classification
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score  # For evaluating the model
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
import numpy as np  # For numerical operations
import warnings  # For suppressing warnings
from imblearn.over_sampling import BorderlineSMOTE  # For handling imbalanced datasets
from imblearn.pipeline import Pipeline as ImbPipeline  # For creating an imbalanced learning pipeline
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only')# Suppress user warnings

# File path
file_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/All_Participants.csv' # Set file path

# Ensure the output directory exists
output_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6' # Set output directory
os.makedirs(output_directory, exist_ok=True) # Create output directory

# Step 1: Data Exploration and Cleaning
# Load the dataset
data = pd.read_csv(file_path) # Load CSV file

# Display the first few rows of the dataset
print("First few rows of the dataset:") # Print header message
print(data.head()) # Display first few rows

# Check for missing values
print("\nMissing values in each column:") # Print missing values header
print(data.isnull().sum()) # Display missing values count

# Drop rows with missing 'State' values, as this is our target variable
data = data.dropna(subset=['State']) # Drop rows with missing target

# Fill missing values in 'Text' column with empty strings
data['Text'] = data['Text'].fillna('') # Fill missing text values

# Display the updated summary
print("\nDataset information after cleaning:") # Print cleaning info header
print(data.info()) # Display data info

# Step 2: Feature Engineering
# Assuming the actual text is in a column named 'Text'
text_feature_column = 'Text' # Define text feature column

# Create a column transformer for handling categorical and numerical features
categorical_features = ['Rule_type', 'Rule_order'] # Define categorical features
numerical_features = ['Updated_True_Time'] # Define numerical features

# TF-IDF transformation for text features
tfidf_vectorizer = TfidfVectorizer() # Initialize TF-IDF vectorizer
text_transformer = Pipeline(steps=[
    ('tfidf', tfidf_vectorizer) # Create text transformer pipeline
])

# Preprocessing pipeline for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features), # Encode categorical features
        ('txt', text_transformer, text_feature_column) # Transform text features
    ],
    remainder='drop' # Drop unused columns
)

# Prepare the features and target variable
X = data.drop(columns=['State']) # Define feature set
y = data['State'] # Define target variable

# Fit the TF-IDF vectorizer on the entire dataset to ensure consistency
X_tfidf = tfidf_vectorizer.fit_transform(X[text_feature_column]) # Fit TF-IDF vectorizer

# Initialize BorderlineSMOTE for more sophisticated oversampling
borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=2) # Initialize BorderlineSMOTE

# Create a balanced random forest classifier with custom class weight adjustment
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced_subsample') # Initialize random forest

# Create a GridSearchCV object with StratifiedKFold
param_grid = {
    'n_estimators': [100, 200, 300], # Define parameter grid
    'max_depth': [None, 10, 20, 30], # Define parameter grid
    'min_samples_split': [2, 5, 10] # Define parameter grid
}

# Set up StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Initialize cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2) # Initialize grid search

# Create an imbalanced pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor), # Add preprocessor to pipeline
    ('borderline_smote', borderline_smote), # Add BorderlineSMOTE to pipeline
    ('classifier', grid_search) # Add classifier to pipeline
])

# Fit the grid search to the data
print("\nStarting grid search...") # Print grid search start message
pipeline.fit(X, y) # Fit pipeline

# Get the best parameters
best_params = pipeline.named_steps['classifier'].best_params_ # Get best parameters
print(f"\nBest parameters: {best_params}") # Print best parameters

# Train the model with the best parameters
rf_classifier_optimized = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced_subsample') # Initialize optimized random forest

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Initialize cross-validation

# Store results of each fold
all_predictions = [] # Initialize predictions list
accuracy_scores = [] # Initialize accuracy scores list
balanced_accuracy_scores = [] # Initialize balanced accuracy scores list
precision_scores = [] # Initialize precision scores list
f1_scores = [] # Initialize F1 scores list

# Check which features contribute most to the predictions
feature_importances = [] # Initialize feature importances list

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # Split data
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Split target
    
    # Preprocess and oversample
    X_train_transformed = preprocessor.fit_transform(X_train) # Transform training data
    X_test_transformed = preprocessor.transform(X_test) # Transform test data
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_transformed, y_train) # Oversample training data
    
    rf_classifier_optimized.fit(X_train_resampled, y_train_resampled) # Fit model
    y_pred = rf_classifier_optimized.predict(X_test_transformed) # Predict on test data
    
    # Save fold predictions
    fold_predictions_df = pd.DataFrame({
        'Actual': y_test, # Actual values
        'Predicted': y_pred # Predicted values
    })
    
    all_predictions.append(fold_predictions_df) # Append fold predictions
    
    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred) # Compute accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred) # Compute balanced accuracy
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) # Compute precision
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) # Compute F1 score
    
    accuracy_scores.append(acc) # Append accuracy
    balanced_accuracy_scores.append(bal_acc) # Append balanced accuracy
    precision_scores.append(prec) # Append precision
    f1_scores.append(f1) # Append F1 score
    
    # Collect feature importances
    feature_importances.append(rf_classifier_optimized.feature_importances_) # Append feature importances
    
    # Print evaluation metrics
    print(f"\nFold {fold+1} evaluation:") # Print fold evaluation header
    print(f"Accuracy: {acc}") # Print accuracy
    print(f"Balanced Accuracy: {bal_acc}") # Print balanced accuracy
    print(f"Precision: {prec}") # Print precision
    print(f"F1 Score: {f1}") # Print F1 score
    print(classification_report(y_test, y_pred, zero_division=0)) # Print classification report

# Combine all fold predictions
combined_predictions = pd.concat(all_predictions, axis=0) # Combine predictions

# Save combined detailed predictions to a single CSV file
combined_predictions.to_csv(f'{output_directory}/Combined_Detailed_Predictions.csv', index=False) # Save predictions to CSV

# Print average metrics across all folds
print("\nAverage metrics across all folds:") # Print average metrics header
print(f"Average Accuracy: {np.mean(accuracy_scores)}") # Print average accuracy
print(f"Average Balanced Accuracy: {np.mean(balanced_accuracy_scores)}") # Print average balanced accuracy
print(f"Average Precision: {np.mean(precision_scores)}") # Print average precision
print(f"Average F1 Score: {np.mean(f1_scores)}") # Print average F1 score

# Handle varying feature importance array sizes
max_length = max(len(f) for f in feature_importances) # Get max array length
padded_importances = [np.pad(f, (0, max_length - len(f)), 'constant') for f in feature_importances] # Pad arrays

# Average feature importances
avg_feature_importances = np.mean(np.vstack(padded_importances), axis=0) # Compute average importances
print("\nAverage Feature Importances:") # Print importances header
print(avg_feature_importances) # Print importances

# Identify the top contributing features
tfidf_length = text_transformer.named_steps['tfidf'].vocabulary_.__len__() # Get TF-IDF length
tfidf_feature_importance = avg_feature_importances[:tfidf_length] # Get TF-IDF importances
other_feature_importance = avg_feature_importances[-len(numerical_features + categorical_features):] # Get other importances

print(f"\nTF-IDF Feature Importance: {np.mean(tfidf_feature_importance)}") # Print TF-IDF importance
print(f"Other Features Importance: {np.mean(other_feature_importance)}") # Print other importance
