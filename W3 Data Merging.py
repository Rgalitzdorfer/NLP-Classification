#Import Libraries
import pandas as pd
import os
from datetime import timedelta

# Function to convert timestamps to seconds
def convert_to_seconds(timestamp):
    time_parts = list(map(int, timestamp.split(':')))
    return timedelta(minutes=time_parts[0], seconds=time_parts[1]).total_seconds()

# Function to incorporate step 0 times and recalculate cumulative times
def adjust_step_0_times(data, additional_log):
    adjusted_data = []
    problems = data['ProblemID'].unique()
    cumulative_time = 0
    
    for problem in problems:
        problem_data = data[data['ProblemID'] == problem]
        if problem_data.empty:
            continue
        step_0_time = additional_log[additional_log['Problem'] == problem]['TrueTimeSpentOnStep'].values[0]
        
        # Add a step 0 entry with only the time
        step_0_entry = {
            'Participant': problem_data.iloc[0]['Participant'],
            'Text': '',
            'Timestamp_seconds': problem_data.iloc[0]['Timestamp_seconds'],
            'ProblemID': problem,
            'ProblemStepID': 0,
            'Rule_order': '',
            'State': '',
            'Rule_type': '',
            'Updated_True_Time': step_0_time,
            'Cumulative_Updated_True_Time': cumulative_time + step_0_time
        }
        adjusted_data.append(step_0_entry)
        cumulative_time += step_0_time
        
        # Add the rest of the steps
        for i, row in problem_data.iterrows():
            row_dict = row.to_dict()
            cumulative_time += row_dict['Updated_True_Time']
            row_dict['Cumulative_Updated_True_Time'] = cumulative_time
            adjusted_data.append(row_dict)
    
    adjusted_df = pd.DataFrame(adjusted_data).sort_values(by='Timestamp_seconds').reset_index(drop=True)
    
    return adjusted_df

# Function to label audio data with corresponding problem and problem step using nearest neighbor approach
def label_audio_data(audio_data, adjusted_data):
    for i, audio_row in audio_data.iterrows():
        audio_time = audio_row['Timestamp_seconds']
        closest_step_idx = (adjusted_data['Cumulative_Updated_True_Time'] - audio_time).abs().idxmin()
        
        if pd.isna(adjusted_data.at[closest_step_idx, 'Text']):
            adjusted_data.at[closest_step_idx, 'Text'] = ''
        
        adjusted_data.at[closest_step_idx, 'Text'] += ' ' + str(audio_row['Text'])
        adjusted_data.at[closest_step_idx, 'Timestamp_seconds'] = audio_time

    return adjusted_data

# Paths
additional_log_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Additional Log Data.csv'
merged_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Merged Data'
adjusted_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data'
audio_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants'

# Ensure the directories exist
os.makedirs(adjusted_data_directory, exist_ok=True)

# Load the additional log data
additional_log = pd.read_csv(additional_log_path)

# List all merged data files in the directory
merged_data_files = [os.path.join(merged_data_directory, f) for f in os.listdir(merged_data_directory) if f.startswith('Merged_Data') and f.endswith('.csv')]

# Process each merged data file
for file_path in merged_data_files:
    try:
        # Load the merged data
        merged_data = pd.read_csv(file_path)
        if merged_data.empty:
            print(f"File {file_path} is empty. Skipping.")
            continue
        print(f"Processing file: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    try:
        # Adjust the data
        adjusted_data = adjust_step_0_times(merged_data, additional_log)
    except Exception as e:
        print(f"Error adjusting data for file {file_path}: {e}")
        continue

    try:
        # Load the corresponding audio data
        file_name = os.path.basename(file_path)
        participant_id = file_name.split('_')[2].split('.')[0]
        audio_data_path = os.path.join(audio_data_directory, f'Participant_{participant_id}_Data.csv')
        audio_data = pd.read_csv(audio_data_path)

        # Ensure the Timestamp_seconds column exists
        if 'Timestamp_seconds' not in audio_data.columns:
            audio_data['Timestamp_seconds'] = audio_data['Timestamp'].apply(convert_to_seconds)

        # Initialize the Text column as an empty string in adjusted data
        if 'Text' not in adjusted_data.columns:
            adjusted_data['Text'] = ''

        # Label the audio data
        final_merged_data = label_audio_data(audio_data, adjusted_data)

        # Save the merged final data back to CSV
        output_path = os.path.join(adjusted_data_directory, file_name.replace('Merged_Data', 'Final_Merged_Data'))
        final_merged_data.to_csv(output_path, index=False)
        print(f"Final merged data saved to {output_path}")
    except Exception as e:
        print(f"Error processing audio data for participant {participant_id}: {e}")
