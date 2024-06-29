#Import Libraries
import pandas as pd #DataFrame Manipulation

#Directories
behavioral_data_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/Behavioral Data.csv'  
all_participants_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5/All_Participants.csv' 
output_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 6/All_Participants_Updated.csv' 

#Read CSVs
behavioral_data = pd.read_csv(behavioral_data_path)  
all_participants = pd.read_csv(all_participants_path)  

#Error Detection
print("Initial Behavioral_Data:")  
print(behavioral_data.head())  
print("\nInitial All_Participants:")  
print(all_participants.head())  

#Match Participant Data
behavioral_data.rename(columns={  
    'Problem': 'ProblemID',  
    'ProblemStep': 'ProblemStepID',  
    'Participant_ID': 'Participant'  
}, inplace=True)  

#Data Standardization
behavioral_data['Participant'] = behavioral_data['Participant'].str.replace('thinkaloudp', '')  
behavioral_data['Participant'] = behavioral_data['Participant'].str.lstrip('0') 
behavioral_data['Participant'] = behavioral_data['Participant'].astype(str)  
all_participants['Participant'] = all_participants['Participant'].astype(str)  

#Error Detection
print("\nBehavioral_Data After Renaming & Stripping:")  
print(behavioral_data.head())  

#Select Columns From Behavioral Data
selected_columns = ["Participant", "ProblemID", "ProblemStepID", "Correct", "First.Action", "Attempt.Count", "NormalizedFirstRT"]  
behavioral_data_selected = behavioral_data[selected_columns]  
print("\nSelected Columns From Behavioral_Data:")  
print(behavioral_data_selected.head()) 

#Merge 
merged_df = pd.merge(all_participants, behavioral_data_selected, on=["Participant", "ProblemID", "ProblemStepID"], how='left')  
print("\nMerged Dataframe:")  # Print merged dataframe header
print(merged_df.head())  # Show merged dataframe

#Save DataFrame
merged_df.to_csv(output_path, index=False) 
print(f"\nMerged dataframe saved to {output_path}")  
