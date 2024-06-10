#Import Libraries
import pandas as pd #DataFrames
import os #Operating System
from datetime import timedelta #Timestamps

#Convert Timestamps to Seconds
def convert_to_seconds(timestamp):
    time_parts = list(map(int, timestamp.split(':')))
    return timedelta(minutes=time_parts[0], seconds=time_parts[1]).total_seconds()

#Incorporate Step 0 Times & Recalculate Cumulative Times
def adjust_step_0_times(data, additional_log):
    adjusted_data = [] #Initialize
    problems = data['ProblemID'].unique() #Get Unique Problem Numbers
    cumulative_time = 0 #Start time at 0
    #Go through all Problems
    for problem in problems:
        problem_data = data[data['ProblemID'] == problem]
        if problem_data.empty:
            continue
        step_0_time = additional_log[additional_log['Problem'] == problem]['TrueTimeSpentOnStep'].values[0] #Fetch from additional log data
        
        # Add a step 0 entry with only the time
        step_0_entry = {
            'Participant': problem_data.iloc[0]['Participant'],
            'Text': '', #Empty Data
            'Timestamp_seconds': problem_data.iloc[0]['Timestamp_seconds'],
            'ProblemID': problem,
            'ProblemStepID': 0,
            'Rule_order': '', #Empty Data
            'State': '', #Empty Data
            'Rule_type': '', #Empty Data
            'Updated_True_Time': step_0_time,
            'Cumulative_Updated_True_Time': cumulative_time + step_0_time
        }
        adjusted_data.append(step_0_entry) #Add to DataFrame
        cumulative_time += step_0_time #Include in running total
        for i, row in problem_data.iterrows():
            row_dict = row.to_dict()
            cumulative_time += row_dict['Updated_True_Time']
            row_dict['Cumulative_Updated_True_Time'] = cumulative_time
            adjusted_data.append(row_dict)
    adjusted_df = pd.DataFrame(adjusted_data).sort_values(by='Timestamp_seconds').reset_index(drop=True) #Sort Values by TimeStamps
    return adjusted_df

#Correlate Time for Text & Log Data 
def label_audio_data(audio_data, adjusted_data):
    for i, row in adjusted_data.iterrows():
        cumulative_time = row['Cumulative_Updated_True_Time'] #Get Cumulative time for current row
        closest_audio_idx = (audio_data['Timestamp_seconds'] - cumulative_time).abs().idxmin() #Find closest audio timestamp
        closest_audio = audio_data.loc[closest_audio_idx] #Get closest audio entry
        adjusted_data.at[i, 'Timestamp_seconds'] = closest_audio['Timestamp_seconds'] #Update Timestamp in new DataFrame
        adjusted_data.at[i, 'Text'] = closest_audio['Text'] #Update Text in new DataFrame
    return adjusted_data

#Directories
additional_log_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Additional Log Data.csv'
merged_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Merged Data'
adjusted_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data'
audio_data_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants'
os.makedirs(adjusted_data_directory, exist_ok=True) #Ensure directories exist
additional_log = pd.read_csv(additional_log_path) #Read additional log data (Problem Step 0)
merged_data_files = [os.path.join(merged_data_directory, f) for f in os.listdir(merged_data_directory) if f.startswith('Merged_Data') and f.endswith('.csv')] #Read all CSV's in folder

#Process & Make Changes to Each Merged Data File
for file_path in merged_data_files:
    try: #Error Detection
        merged_data = pd.read_csv(file_path) #Load Merged Data
        if merged_data.empty:
            print(f"File {file_path} is empty. Skipping.")
            continue
        print(f"Processing file: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue
    try: #Error Detection
        adjusted_data = adjust_step_0_times(merged_data, additional_log) #Add Step 0
    except Exception as e:
        print(f"Error adjusting data for file {file_path}: {e}")
        continue
    try: #Error Detection
        file_name = os.path.basename(file_path)
        participant_id = file_name.split('_')[2].split('.')[0]
        audio_data_path = os.path.join(audio_data_directory, f'Participant_{participant_id}_Data.csv')
        audio_data = pd.read_csv(audio_data_path)
        if 'Timestamp_seconds' not in audio_data.columns: #Ensure TimeStamp Column Exists
            audio_data['Timestamp_seconds'] = audio_data['Timestamp'].apply(convert_to_seconds) #Make Conversion
        if 'Text' not in adjusted_data.columns: #Initialize
            adjusted_data['Text'] = ''
        final_merged_data = label_audio_data(audio_data, adjusted_data) #Correlate Times
        #Save DataFrame
        output_path = os.path.join(adjusted_data_directory, file_name.replace('Merged_Data', 'Final_Merged_Data'))
        final_merged_data.to_csv(output_path, index=False)
        print(f"Final Merged Data Saved to {output_path}")
    except Exception as e: #Error Detection
        print(f"Error processing audio data for participant {participant_id}: {e}")
