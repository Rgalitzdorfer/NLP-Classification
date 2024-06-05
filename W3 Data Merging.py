#Import Libraries
import pandas as pd
import os

# Function to incorporate step 0 times and recalculate cumulative times
def adjust_step_0_times(data, additional_log):
    adjusted_data = []
    problems = data['ProblemID'].unique()
    cumulative_time = 0
    
    for problem in problems:
        problem_data = data[data['ProblemID'] == problem]
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

# Function to label audio data with corresponding problem and problem step
def label_audio_data(audio_data, adjusted_data):
    labeled_audio_data = []

    for i, audio_row in audio_data.iterrows():
        audio_time = audio_row['Timestamp_seconds']
        for j, step_row in adjusted_data.iterrows():
            step_start_time = step_row['Timestamp_seconds']
            step_end_time = step_row['Cumulative_Updated_True_Time']
            if step_start_time <= audio_time <= step_end_time:
                labeled_audio_row = audio_row.copy()
                labeled_audio_row['ProblemID'] = step_row['ProblemID']
                labeled_audio_row['ProblemStepID'] = step_row['ProblemStepID']
                labeled_audio_data.append(labeled_audio_row)
                break

    labeled_audio_df = pd.DataFrame(labeled_audio_data)
    return labeled_audio_df

# Paths
additional_log_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Additional Log Data.csv'
merged_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Merged Data'
adjusted_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data'
audio_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants'
labeled_audio_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data/Labeled Audio Data'

# Ensure the directories exist
os.makedirs(adjusted_data_directory, exist_ok=True)
os.makedirs(labeled_audio_data_directory, exist_ok=True)

# Load the additional log data
additional_log = pd.read_csv(additional_log_path)

# List all merged data files in the directory
merged_data_files = [os.path.join(merged_data_directory, f) for f in os.listdir(merged_data_directory) if f.startswith('Merged_Data') and f.endswith('.csv')]

# Process each merged data file
for file_path in merged_data_files:
    try:
        # Load the merged data
        merged_data = pd.read_csv(file_path)
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
        # Save the adjusted data back to CSV
        file_name = os.path.basename(file_path)
        output_path = os.path.join(adjusted_data_directory, file_name.replace('Merged_Data', 'Final_Merged_Data'))
        adjusted_data.to_csv(output_path, index=False)
        print(f"Adjusted data saved to {output_path}")
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")

    try:
        # Load the corresponding audio data
        participant_id = file_name.split('_')[2].split('.')[0]
        audio_data_path = os.path.join(audio_data_directory, f'Participant_{participant_id}_Data.csv')
        audio_data = pd.read_csv(audio_data_path)

        # Label the audio data
        labeled_audio_data = label_audio_data(audio_data, adjusted_data)

        # Save the labeled audio data to a new CSV file
        labeled_audio_output_path = os.path.join(labeled_audio_data_directory, f'Labeled_Audio_Data_{participant_id}.csv')
        labeled_audio_data.to_csv(labeled_audio_output_path, index=False)
        print(f"Labeled audio data saved to {labeled_audio_output_path}")
    except Exception as e:
        print(f"Error processing audio data for participant {participant_id}: {e}")
