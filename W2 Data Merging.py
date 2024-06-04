#Import Libraries
import pandas as pd #DataFrames
import os #File paths
from datetime import datetime, timedelta #Timestamps
import numpy as np


##Part 1
#Function to process individual file
def process_file(file_path):
    with open(file_path, 'r') as file: #Read mode
        content = file.read().splitlines() #Split content into lines
    participant_id = os.path.basename(file_path).split('.')[0] 
    data = []
    current_time = ""
    current_text = ""
    #Iterate through each line in file
    for line in content: 
        if line.startswith("Unknown Speaker") or line.startswith("Speaker"):
            if current_time and current_text:
                data.append([participant_id, current_time, current_text])
                current_text = ""
            parts = line.split()
            current_time = parts[2]
        else:
            current_text += line.strip() + " "
    if current_time and current_text:
        data.append([participant_id, current_time, current_text])
    return data
#Directories
source_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/NCS_Recordings/'
target_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants/'
os.makedirs(target_directory, exist_ok=True)

#List of participant IDs
participants = ['0115', '0207', '0213', '0216', '0217', '0403', '0405', '0414', '0421', '0501', '0606', '0701', '0802', '1006', '1111', '1203', '1212']
#Process each participant's file and save the dataframe
for participant in participants:
    file_path = os.path.join(source_directory, f'ThinkAloud{participant}.txt')
    if os.path.exists(file_path):
        data = process_file(file_path)
        df = pd.DataFrame(data, columns=['Participant', 'Timestamp', 'Text'])
        output_path = os.path.join(target_directory, f'Participant_{participant}_Data.csv')
        df.to_csv(output_path, index=False)
        print(f"Data for participant {participant} saved to {output_path}")
    else:
        print(f"File for participant {participant} not found.")



##Part 2
#Get Log Data
log_data_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/W1_Log_Data.csv'
log_data = pd.read_csv(log_data_path)
print("Log Data:")
print(log_data.head())
print("Use Cumulative 'Time Spent' & 'True Time Spent' to calculate with verbal data.")

#Get 'Step 0' Log Data
additional_log_data_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Additional Log Data.csv'
additional_log_data = pd.read_csv(additional_log_data_path)

#Convert to seconds
def convert_to_seconds(timestamp):
    time_parts = list(map(int, timestamp.split(':')))
    return timedelta(minutes=time_parts[0], seconds=time_parts[1]).total_seconds()

#Data Cleaning
log_data['Participant_ID'] = log_data['Participant_ID'].str.lower()
log_data = log_data[log_data['Participant_ID'] != 'thinkaloudp0101']
log_data = log_data.sort_values(by=['Participant_ID', 'Problem', 'ProblemStep'])
participants = log_data['Participant_ID'].str.extract(r'(\d+)')[0].unique()

#File paths
participant_files_base_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants'
merged_files_base_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Merged Data/'

#Iterate over each participant
for participant_id in participants:
    participant_file_path = os.path.join(participant_files_base_path, f'Participant_{participant_id}_Data.csv') #Define file path for each participant
    if not os.path.exists(participant_file_path): #Check if file exists
        print(f"File for Participant {participant_id} not found. Skipping.")
        continue
    participant_data = pd.read_csv(participant_file_path) #Load participant data
    participant_data['Timestamp_seconds'] = participant_data['Timestamp'].apply(convert_to_seconds) #Convert timestamps to seconds
    participant_data['Participant'] = participant_data['Participant'].str.lower().str.replace("thinkaloud", "thinkaloudp") #Normalize participant IDs
    
    current_participant = f"thinkaloudp{participant_id}" #Format current participant ID
    participant_log = log_data[log_data['Participant_ID'] == current_participant].copy() #Filter log data for current participant
    additional_log = additional_log_data[additional_log_data['Participant_ID'] == current_participant] #Filter additional log data
    if participant_log.empty: #Check if log data is empty
        print(f"No log data for Participant {participant_id}. Skipping.")
        continue
    if additional_log.empty: #Check if additional log data is empty
        print(f"No additional log data for Participant {participant_id}. Skipping.")
        continue

    print(f"\nLog Data for Participant {participant_id}:")
    print(participant_log[['Problem', 'ProblemStep']].head()) #Print first few rows of log data
    print(f"\nAdditional Log Data for Participant {participant_id}:")
    print(additional_log[['Problem', 'TrueTimeSpentOnStep']].head()) #Print first few rows of additional log data
    #Merge logs and include all problems
    merged_logs = pd.merge(participant_log, additional_log[['Participant_ID', 'Problem', 'TrueTimeSpentOnStep']], 
                           on=['Participant_ID', 'Problem'], how='outer', suffixes=('', '_additional')) #Merge log data
    merged_logs.rename(columns={'TrueTimeSpentOnStep_additional': 'Updated_True_Time'}, inplace=True) #Rename column
    #Fill missing values
    if 'TrueTimeSpentOnStep' in merged_logs.columns: #Check if column exists
        merged_logs['Updated_True_Time'] = merged_logs['Updated_True_Time'].fillna(merged_logs['TrueTimeSpentOnStep']) #Fill missing values
    #Add missing problems explicitly
    for problem in additional_log['Problem'].unique(): #Iterate over unique problems
        if problem not in merged_logs['Problem'].unique(): #Check if problem is missing
            missing_problem_row = additional_log[additional_log['Problem'] == problem].iloc[0] #Get missing problem row
            new_row = {
                'Participant_ID': missing_problem_row['Participant_ID'],
                'Problem': missing_problem_row['Problem'],
                'ProblemStep': 0,
                'Rule_order': 'Unknown',
                'State': 'Unknown',
                'Rule_type': 'Unknown',
                'Updated_True_Time': missing_problem_row['TrueTimeSpentOnStep'],
                'TimeSpent': missing_problem_row['TrueTimeSpentOnStep']
            }
            merged_logs = merged_logs.append(new_row, ignore_index=True) #Append missing problem row
    merged_logs = merged_logs.sort_values(by=['Problem', 'ProblemStep']).reset_index(drop=True) #Sort merged logs
    participant_data = participant_data.sort_values(by='Timestamp_seconds').reset_index(drop=True) #Sort participant data
    merged_data = []
    for i, row in participant_data.iterrows(): #Iterate over participant data
        merged_entry = row.to_dict()
        log_row = merged_logs.iloc[i % len(merged_logs)] #Get corresponding log row
        merged_entry['ProblemID'] = log_row['Problem']
        merged_entry['ProblemStepID'] = log_row['ProblemStep']
        merged_entry['Rule_order'] = log_row['Rule_order']
        merged_entry['State'] = log_row['State']
        merged_entry['Rule_type'] = log_row['Rule_type']
        merged_entry['Updated_True_Time'] = log_row['Updated_True_Time']
        merged_data.append(merged_entry) #Append merged entry
    #Create DataFrame
    merged_df = pd.DataFrame(merged_data)  
    merged_df['Cumulative_Updated_True_Time'] = merged_df['Updated_True_Time'].cumsum() #Calculate cumulative time
    time_columns_to_keep = ['Timestamp_seconds', 'Updated_True_Time', 'Cumulative_Updated_True_Time', 'ProblemID', 'ProblemStepID', 'Rule_order', 'State', 'Rule_type']
    columns_to_drop = [col for col in merged_df.columns if 'Time' in col and col not in time_columns_to_keep]
    merged_df.drop(columns=columns_to_drop, inplace=True) #Drop unnecessary columns
    merged_file_path = os.path.join(merged_files_base_path, f'Merged_Data_{participant_id}.csv') #Define file path for saving
    merged_df.to_csv(merged_file_path, index=False)  
    print(f"Data for Participant {participant_id} Merged & Saved Successfully.")  