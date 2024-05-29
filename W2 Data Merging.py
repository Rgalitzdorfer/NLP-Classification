#Import Libraries
import pandas as pd #DataFrames
import os #File paths
from datetime import datetime, timedelta #Timestamps


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

#Convert timestamps to seconds
def convert_to_seconds(timestamp):
    time_parts = list(map(int, timestamp.split(':')))
    return timedelta(minutes=time_parts[0], seconds=time_parts[1]).total_seconds()

#Data Cleaning
log_data['Participant_ID'] = log_data['Participant_ID'].str.lower() #Make all lowercase 
log_data = log_data[log_data['Participant_ID'] != 'thinkaloudp0101'] #Remove participant with inconsistent verbal data

#Sort Values
log_data = log_data.sort_values(by=['Participant_ID', 'Problem', 'ProblemStep']) #Problem & Problem Step follow order
print("Log Data After Sorting:") #Error Detection
print(log_data[['TimeSpent', 'TrueTimeSpentOnStep', 'Problem', 'ProblemStep']].head(20)) #Error Detection

#File Directories
participant_files_base_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Participants' #Read Text Data
merged_files_base_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 2/Merged Data/' #Save to Merged Data

#Iterate over each participant
for participant_id in participants:
    participant_file_path = os.path.join(participant_files_base_path, f'Participant_{participant_id}_Data.csv') #Save each participant
    if not os.path.exists(participant_file_path): #Error Detection
        print(f"File for Participant {participant_id} not found. Skipping.")
        continue
    participant_data = pd.read_csv(participant_file_path) #Read DataFrame
    participant_data['Timestamp_seconds'] = participant_data['Timestamp'].apply(convert_to_seconds) #Make Timestamp Seconds column for convenience
    participant_data['Participant'] = participant_data['Participant'].str.lower().str.replace("thinkaloud", "thinkaloudp") #Normalize Data 
    merged_data = [] #Initialize
    current_participant = f"thinkaloudp{participant_id}" #Do one participant at a time
    participant_log = log_data[log_data['Participant_ID'] == current_participant].copy() #Copy method to avoid slice warnings
    if participant_log.empty: #Error Detection
        print(f"No log data for Participant {participant_id}. Skipping.")
        continue
    #Print Statements to validate code
    print(f"\nLog Data for Participant {participant_id}:")
    print(participant_log[['TimeSpent', 'TrueTimeSpentOnStep', 'Problem', 'ProblemStep']].head())

    #Merge Data 
    for i, row in participant_data.iterrows():
        if i >= len(participant_log): #Error Detection
            break
        log_row = participant_log.iloc[i]
        merged_entry = row.to_dict()
        merged_entry['ProblemID'] = log_row['Problem']
        merged_entry['ProblemStepID'] = log_row['ProblemStep']
        merged_entry['Rule_order'] = log_row['Rule_order']
        merged_entry['State'] = log_row['State']
        merged_entry['Rule_type'] = log_row['Rule_type']
        merged_entry['TimeSpent'] = log_row['TimeSpent']
        merged_entry['TrueTimeSpentOnStep'] = log_row['TrueTimeSpentOnStep']
        merged_data.append(merged_entry)
    #Create DataFrame
    merged_df = pd.DataFrame(merged_data) 
    merged_df['Cumulative_TimeSpent'] = merged_df['TimeSpent'].cumsum() #Calculate running total for TimeSpent
    #Calculate running total for TrueTimeSpentOnStep only adding unique values (to avoid double counting)
    cumulative_true_time = 0
    cumulative_true_times = []
    last_seen = None
    for value in merged_df['TrueTimeSpentOnStep']:
        if value != last_seen:
            cumulative_true_time += value
            last_seen = value
        cumulative_true_times.append(cumulative_true_time)
    merged_df['Cumulative_TrueTimeSpentOnStep'] = cumulative_true_times #Create Column

    #Save DataFrame
    merged_file_path = os.path.join(merged_files_base_path, f'Merged_Data_{participant_id}.csv')
    merged_df.to_csv(merged_file_path, index=False)
    print(f"Data for Participant {participant_id} Merged & Saved Successfully.")
