#Import Libraries
import os
import pandas as pd

#Directories
input_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Text Distribution'
output_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 5'
output_file = os.path.join(output_folder, 'All_Participants.csv')

#Initialize DataFrame
dfs = []

#Read All Individual Participant Files
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

#Concatenate to One DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

#Data Cleaning
merged_df['Participant'] = merged_df['Participant'].str.replace('thinkaloudp', '').astype(int)
merged_df['ProblemID'] = merged_df['ProblemID'].astype('category')
merged_df['ProblemStepID'] = merged_df['ProblemStepID'].astype('category')
merged_df = merged_df.sort_values(by=['Participant', 'ProblemID', 'ProblemStepID']) #Ensure Correct Sorting

#Remove Useless States
def remove_phrases(state):
    phrases_to_remove = ['Reading_question', 'Relation_to_goal']
    for phrase in phrases_to_remove:
        state = state.replace(phrase, '')
    return state.strip()  
if 'State' in merged_df.columns: #Apply Function
    merged_df['State'] = merged_df['State'].apply(remove_phrases) 

#Create 'Label' Column
def create_labels(state): 
    label_mapping = {
        'Rule_following': 'RF',
        'Rule_discovery': 'RD',
        'Rule_search': 'RS',
        'Follow_wrong': 'FW',
        'Rule_violation': 'RV'
    }
    labels = [label_mapping[key] for key in label_mapping if key in state]
    labels = sorted(labels)
    return ','.join(labels)
merged_df['Label'] = merged_df['State'].apply(create_labels) #Apply Function

#Keep Labels With 3+ Occurences
labels_to_keep = ['RF', 'FW,RD,RS', 'RD,RF,RS,RV', 'RD,RS']
filtered_df = merged_df[merged_df['Label'].isin(labels_to_keep)] 
filtered_df.to_csv(output_file, index=False) #Save to CSV

#Descriptive Statistics
label_counts = filtered_df['Label'].value_counts()
print("Unique label combinations and their counts:")
for label, count in label_counts.items():
    print(f"'{label}': {count} times")
print(f"\nTotal Unique Combinations: {len(label_counts)}")
print(f"Total Records Processed: {filtered_df.shape[0]}")
print(f"All Files are Merged, Cleaned, Labeled, & Filtered. Saved to {output_file}")


#Made One DataFrame for All Participants.
#Deleted 'Reading Question' & 'Relation to Goal' from State Column.
#Created Labels for Each Cognitive State.
#Effectively Removed Problem Step 0 Rows as They Had ' ' for Cognitive State.
#Deleted All Labels With Less Than 3 Occurences (4 Remaining). 
 