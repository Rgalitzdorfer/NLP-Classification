import pandas as pd  # Import pandas

# File paths
behavioral_data_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/Behavioral Data.csv'  # Path to behavioral data
all_participants_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/All_Participants.csv'  # Path to participants data
output_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv'  # Output file path

# Load the data
behavioral_data = pd.read_csv(behavioral_data_path)  # Load behavioral data
all_participants = pd.read_csv(all_participants_path)  # Load participants data

# Print initial dataframes
print("Initial behavioral_data:")  # Print initial behavioral data header
print(behavioral_data.head())  # Show first few rows of behavioral data
print("\nInitial all_participants:")  # Print initial participants data header
print(all_participants.head())  # Show first few rows of participants data

# Rename columns in behavioral data to match all participants data
behavioral_data.rename(columns={  # Rename columns
    'Problem': 'ProblemID',  # Rename Problem to ProblemID
    'ProblemStep': 'ProblemStepID',  # Rename ProblemStep to ProblemStepID
    'Participant_ID': 'Participant'  # Rename Participant_ID to Participant
}, inplace=True)  # Apply changes

# Strip the prefix "thinkaloudp" from participant IDs in behavioral data
behavioral_data['Participant'] = behavioral_data['Participant'].str.replace('thinkaloudp', '')  # Remove prefix

# Remove leading zeros from Participant IDs in behavioral data
behavioral_data['Participant'] = behavioral_data['Participant'].str.lstrip('0')  # Remove leading zeros

# Ensure both 'Participant' columns are of the same type (string)
behavioral_data['Participant'] = behavioral_data['Participant'].astype(str)  # Convert to string
all_participants['Participant'] = all_participants['Participant'].astype(str)  # Convert to string

# Print after renaming and stripping
print("\nBehavioral_data after renaming and stripping:")  # Print after modifications header
print(behavioral_data.head())  # Show modified behavioral data

# Select the specified columns from behavioral data
selected_columns = ["Participant", "ProblemID", "ProblemStepID", "Correct", "First.Action", "Attempt.Count", "NormalizedFirstRT"]  # Columns to select
behavioral_data_selected = behavioral_data[selected_columns]  # Select columns

# Print selected columns from behavioral_data
print("\nSelected columns from behavioral_data:")  # Print selected columns header
print(behavioral_data_selected.head())  # Show selected columns

# Merge the dataframes
merged_df = pd.merge(all_participants, behavioral_data_selected, on=["Participant", "ProblemID", "ProblemStepID"], how='left')  # Merge dataframes

# Print merged dataframe
print("\nMerged dataframe:")  # Print merged dataframe header
print(merged_df.head())  # Show merged dataframe

# Save the merged dataframe
merged_df.to_csv(output_path, index=False)  # Save to CSV

print(f"\nMerged dataframe saved to {output_path}")  # Confirm save
