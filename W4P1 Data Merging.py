#Import Libraries
import os #Operating Systems
import pandas as pd #DataFrames

#Directories
input_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 3/Final Merged Data' #Input
output_folder = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 4/Complete Data' #Output
os.makedirs(output_folder, exist_ok=True) #Ensure they exist

#Process each file
def process_file(file_path):
    df = pd.read_csv(file_path)
    print(f"Processing {file_path}") #Error Detection
    print(f"Columns: {df.columns.tolist()}") #Error Detection
    problem_col = 'ProblemID' 
    step_col = 'ProblemStepID'
    df['Text'] = df['Text'].astype(str).apply(lambda x: ' '.join(x.split())) #Convert to string & remove extra spaces
    if 'State' in df.columns:
        df['State'] = df['State'].fillna(' ') #Fill Nan values
    if 'Rule_order' in df.columns:
        df['Rule_order'] = df['Rule_order'].fillna(' ') #Fill Nan values
    timestamp_columns = ['Cumulative_Updated_True_Time', 'Timestamp_seconds'] #Column Identification
    text_columns = ['Text', 'State'] #Column Identification
    updated_true_time_column = 'Updated_True_Time' #Column Identification
    columns_to_remove_duplicates = [col for col in df.columns if col not in text_columns + timestamp_columns + [updated_true_time_column, problem_col, step_col]] #Remove Duplicates for Some Columns
    grouped = df.groupby([problem_col, step_col]) #Group by Problem & Problem Step
    #Concatenate Text
    def concatenate_text_state(series):
        return ' '.join(series.dropna().astype(str))
    #Remove Duplicates, Concatenate Values
    def remove_duplicates(series):
        return ', '.join(sorted(set(series.dropna().astype(str))))
    #Trim at '||' Delimiter
    def concatenate_and_remove_duplicates(text_series):
        texts = text_series.dropna().unique()
        concatenated_text = ' || '.join(texts)
        return concatenated_text.split(' || ')[0] #Trim at the first occurence
    #Aggregate the data
    aggregation_dict = {
        'Text': concatenate_and_remove_duplicates,
        'State': concatenate_text_state if 'State' in df.columns else 'first',
        'Cumulative_Updated_True_Time': 'last',
        'Timestamp_seconds': 'last',
        'Rule_order': lambda x: ', '.join(sorted(set(x.dropna().astype(str)))),
        'Updated_True_Time': lambda x: round(x.dropna().astype(float).sum(), 2), #Round to 2 Decimal Places
        'Rule_type': lambda x: ', '.join(sorted(set(x.dropna().astype(str)))),
        'Participant': 'first'
    }
    #Add columns to aggregation dictionary
    for col in columns_to_remove_duplicates:
        if col in df.columns:
            aggregation_dict[col] = remove_duplicates
    aggregated_df = grouped.agg(aggregation_dict).reset_index() #Reset Index to avoid incorrect order
    return aggregated_df


#Process Each CSV File in Input Folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        try:
            transformed_df = process_file(file_path)
            participant_id = transformed_df['Participant'].iloc[0]
            participant_number = ''.join(filter(str.isdigit, participant_id)) 
            output_filename = f"Complete_Data_{participant_number}.csv"
            output_file_path = os.path.join(output_folder, output_filename)
            transformed_df.to_csv(output_file_path, index=False)
            print(f'\nTransformed data saved to {output_file_path}')
        except KeyError as e: #Error Detection
            print(f"Skipping {filename} due to missing column: {e}")
        except ValueError as e: #Error Detection
            print(f"Skipping {filename} due to ValueError: {e}")
        except Exception as e: #Error Detection
            print(f"Skipping {filename} due to unexpected error: {e}")
print("Data Transformation Complete.")