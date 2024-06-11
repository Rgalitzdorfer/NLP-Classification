#Import Libraries 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Read CSV
log_data = pd.read_csv('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/W1_Log_Data.csv')
print(log_data.head()) #Ensure Proper DataFrame

##Task 1: Summary Stats (Mean, Median, Distribution)
mean_values = log_data[['TimeSpent', 'TrueTimeSpentOnStep', 'Pre_Test', 'Post_Test', 'Gain', 'n_gain']].mean() 
median_values = log_data[['TimeSpent', 'TrueTimeSpentOnStep', 'Pre_Test', 'Post_Test', 'Gain', 'n_gain']].median()
std_values = log_data[['TimeSpent', 'TrueTimeSpentOnStep', 'Pre_Test', 'Post_Test', 'Gain', 'n_gain']].std()
summary_stats = pd.DataFrame({ #Make DataFrame
    'Mean': mean_values,
    'Median': median_values,
    'Standard_Deviation': std_values
})
print("TASK 1 Takeaway: Summary Statistics:")
print(summary_stats)
summary_stats.to_csv('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T1Summary_Stats.csv')



##Task 2: Distributuon With Outliers
#Calculate Outliers with IQR
def identify_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return lower_bound, upper_bound, outliers
#Columns/Titles to Plot
columns = ['TimeSpent', 'TrueTimeSpentOnStep', 'Pre_Test', 'Post_Test', 'Gain', 'n_gain']
titles = ['Time Spent', 'True Time Spent On Step', 'Pre Test', 'Post Test', 'Gain', 'Normalized Gain']
#Save Plots
output_dir = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T2Outliers_Plots'
os.makedirs(output_dir, exist_ok=True)
#Create Histograms
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
outlier_dataframes = []
#Loop through all columns
for ax, column, title in zip(axes.flatten(), columns, titles):
    sns.histplot(log_data[column].dropna(), bins=30, edgecolor='black', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    lower_bound, upper_bound, outliers = identify_outliers(log_data, column)
    ax.axvline(lower_bound, color='red', linestyle='--')
    ax.axvline(upper_bound, color='red', linestyle='--')
    outliers = outliers.copy()
    outliers['Outlier_Column'] = column
    outlier_dataframes.append(outliers)
plt.tight_layout()
plot_filename = os.path.join(output_dir, 'Histograms_With_Outliers.png') #Save histograms
plt.savefig(plot_filename)
plt.close()
print("TASK 2 Takeaway: True time spent & time spent had the most amount of outliers.")


##Task 3: Average Time Spent On Rule Types (Per Participant)
participant_ids = log_data['Participant_ID'].unique()
#Directory to save the plots
output_dir = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T3Average_Time_Spent' #Directory to save the plots
os.makedirs(output_dir, exist_ok=True)

#Loop through each Participant_ID and plot the average time spent
for participant_id in participant_ids:
    participant_data = log_data[log_data['Participant_ID'] == participant_id]
    average_time_spent = participant_data.groupby(['Rule_type', 'Rule_order'])['TimeSpent'].mean().reset_index()
    #Plot the average time spent on each Rule type
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=average_time_spent, x='Rule_order', y='TimeSpent', hue='Rule_type', marker='o')
    plt.title(f'Average Time Spent on Each Rule Type for Participant {participant_id}')
    plt.xlabel('Rule Order')
    plt.ylabel('Average Time Spent')
    plt.legend(title='Rule Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'Average_Time_Spent_on_Rule_Type_{participant_id}.png')
    plt.savefig(plot_filename)
    plt.close()
print("TASK 3 Takeaway: As a general rule, an increase in the amount a participant saw a rule (rule order) led to a decrease in average time spent.")


##Task 4: Frequency of Unique Cognitive States
state_frequencies = log_data['State'].value_counts().reset_index()
state_frequencies.columns = ['State', 'Frequency']
#Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=state_frequencies, x='State', y='Frequency', palette='viridis', hue='State', legend=False)
plt.title('Frequency of Each Unique State Experienced by Participants')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
#Save Plot
output_plot_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T4State_Frequencies.png'
plt.savefig(output_plot_path)
print("TASK 4 Takeaway: 'Reading question' & 'Rule following' were most frequent cognitive states.")



##Task 5: Total/Average Time on Unique Cognitive States
state_time_metrics = log_data.groupby('State')['TimeSpent'].agg(['sum', 'mean']).reset_index() 
state_time_metrics.columns = ['State', 'TotalTimeSpent', 'AverageTimeSpent']
#Plot Total Time
plt.figure(figsize=(12, 8))
sns.barplot(data=state_time_metrics, x='State', y='TotalTimeSpent', palette='viridis', hue='State', legend=False)
plt.title('Total Time Spent for Each State')
plt.xlabel('State')
plt.ylabel('Total Time Spent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T5Total_Time_Per_State.png')
print("TASK 5 Takeaway: Total time is most spent on 'Rule Following'")

#Plot Average Time
plt.figure(figsize=(12, 8))
sns.barplot(data=state_time_metrics, x='State', y='AverageTimeSpent', palette='viridis', hue='State', legend=False)
plt.title('Average Time Spent for Each State')
plt.xlabel('State')
plt.ylabel('Average Time Spent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T5Average_Time_Per_State.png')
print("TASK 5 Takeaway: Average time is most spent on 'Rule Search'")
#File paths
total_time_spent_plot_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T5Total_Time_Per_State.png'
average_time_spent_plot_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T5Average_Time_Per_State.png'



##Task 6: Investigate Correlation Between Cognitive State & Learning Gains
state_gain_metrics = log_data.groupby('State')['Gain'].agg(['mean', 'std']).reset_index() #Get average & standard deviation
state_gain_metrics.columns = ['State', 'AverageGain', 'GainStdDev'] #Useful Columns
#Plot Average Gain for Unique State
plt.figure(figsize=(12, 8))
sns.barplot(data=state_gain_metrics, x='State', y='AverageGain', palette='viridis', hue='State', legend=False)
plt.title('Average Gain for Each State')
plt.xlabel('State')
plt.ylabel('Average Gain')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T6Average_Gain_Per_State.png')
print("TASK 6 Takeaway: Relation to Goal has lowest average Gain")

#Plot Standard Deviation for Unique State
plt.figure(figsize=(12, 8))
sns.barplot(data=state_gain_metrics, x='State', y='GainStdDev', palette='viridis', hue='State', legend=False)
plt.title('Standard Deviation of Gain for Each State')
plt.xlabel('State')
plt.ylabel('Standard Deviation of Gain')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T6StdDev_Gain_Per_State.png')
print("TASK 6 Takeaway: Lower Standard Deviation = Learning Gains Consistent Amongst Participants")
#File Paths
average_gain_plot_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T6Average_Gain_Per_State.png'
gain_stddev_plot_path = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T6StdDev_Gain_Per_State.png'



##Task 7: Rule Order Impact (Unique Rule Types)
rule_types = log_data['Rule_type'].unique()
save_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T7Rule_Order_Impacts'  
# Ensure the save directory exists
os.makedirs(save_directory, exist_ok=True)
#Iterate Rule Types
for rule_type in rule_types:
    subset = log_data[log_data['Rule_type'] == rule_type]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Rule_order', y='TimeSpent', data=subset, label='TimeSpent', marker='o')
    sns.lineplot(x='Rule_order', y='TrueTimeSpentOnStep', data=subset, label='TrueTimeSpentOnStep', marker='x')
    plt.title(f'TimeSpent and TrueTimeSpentOnStep vs Rule_order for {rule_type}')
    plt.xlabel('Rule_order')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plot_path = os.path.join(save_directory, f'Time_vs_Rule_Order_{rule_type}.png')
    plt.savefig(plot_path)
    plt.close()
print("TASK 7 Takeawy: For 5 of 7 Rule Types, time spent had an inverse relationship with rule order.")



##Task 8: Gain & Normalized Gain
save_directory = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T8Gain_&_Normalized_Gain' 
os.makedirs(save_directory, exist_ok=True)
#Violin Plot (Distribution & Density of Gains)
melted_data = log_data.melt(value_vars=['Gain', 'n_gain'], var_name='Type', value_name='Value') #Fit in one image
plt.figure(figsize=(12, 6))
sns.boxplot(x='Type', y='Value', data=melted_data)
plt.title('Box Plot of Gain and Normalized Gain')
plt.xlabel('Type')
plt.ylabel('Value')
plot_path = os.path.join(save_directory, 'Distribution_of_Gain_&_N_Gain.png')
plt.savefig(plot_path)
plt.close()

#Identify Participants with Highest & Lowest Gains
average_gains = log_data.groupby('Participant_ID')[['Gain', 'n_gain']].mean().reset_index() #Get average for each participant
highest_avg_gain = average_gains.loc[average_gains['Gain'].idxmax()] 
lowest_avg_gain = average_gains.loc[average_gains['Gain'].idxmin()]
highest_avg_ngain = average_gains.loc[average_gains['n_gain'].idxmax()]
lowest_avg_ngain = average_gains.loc[average_gains['n_gain'].idxmin()]
#Save Results via Text File
results_path = os.path.join(save_directory, 'Highest_and_Lowest_Participant_Gains.txt')
with open(results_path, 'w') as f:
    f.write("Participant with Highest Gain:\n")
    f.write(highest_avg_gain.to_string())
    f.write("\n\nParticipant with Lowest Gain:\n")
    f.write(lowest_avg_gain.to_string())
    f.write("\n\nParticipant with Highest Normalized Gain:\n")
    f.write(highest_avg_ngain.to_string())
    f.write("\n\nParticipant with Lowest Normalized Gain:\n")
    f.write(lowest_avg_ngain.to_string())
print("TASK 8 Takeaway: Participant ID with lowest gains: 403")
print("TASK 8 Takeaway: Participant ID with lowest normalized gains: 403")
print("TASK 8 Takeaway: Participant ID with highest gains: 101")
print("TASK 8 Takeaway: Participant ID with highest normalized gains: 216")



##TASK 9: Gain & Timespent in Different Cognitive States
output_dir = '/Users/ryangalitzdorfer/Downloads/FACETLab/Week 1/T9Cognitive_State_Plots'
os.makedirs(output_dir, exist_ok=True)
states = log_data['State'].unique() #Get Unique States
#Iterate for each state
for state in states:
    state_data = log_data[log_data['State'] == state]
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=state_data, x='TimeSpent', y='Gain', hue='State', palette='viridis')
    plt.title(f'Relationship between Gain and TimeSpent for State: {state}')
    plt.xlabel('TimeSpent')
    plt.ylabel('Gain')
    plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #Save Plot
    plot_path = os.path.join(output_dir, f'Relationship_between_Gain_and_TimeSpent_State_{state}.png')
    plt.savefig(plot_path)
    plt.close()
print("TASK 9 Takeaway: A majority of gains occur after only a short amount of time was spent. There is no clear correlation; however, it seems that many results cluster indicating that certain results might yield a higher frequency.")