import os
import pandas as pd
import pinocchio as pin
import numpy as np



#df = pd.read_csv('Data_1/resynchronized_data_subject003.csv')
#print([c for c in df.columns if 'tau' in c.lower()])
#print([c for c in df.columns if 'block' in c.lower() or 'trial' in c.lower()])
#print(df[['block_idx', 'desired_trial_time.0', 'remaining_trials', 'total_trials']].head(10))

#df = pd.read_csv('Data_1/Subject3_processed.csv')
#print(df['block_id'].unique())

#df_b5 = df[df['block_id'] == 5000.0]
#print(f"Total rows: {len(df_b5)}")
#print(f"motor_force unique values: {df_b5['motor_force'].unique()}")

## Check motor force timing
#motor_on = df[df['motor_force'] != 0]
#print(motor_on['time'].diff().describe())   # tells you gap between pulses → gives frequency

# Check CoM lateral range
#print(df['com_y'].describe())   # tells you min/max lateral CoM → gives amplitude


#df = pd.read_csv('Data_1/Subject3_features.csv')
#print(df['motor_force'].describe())
#print("\nUnique non-zero absolute values (top 20):")
#print(df[df['motor_force'].abs() > 0]['motor_force'].abs().sort_values().unique()[:20])

#df = pd.read_csv('Data_1/Subject3_features.csv')
#perturbed = df[df['motor_force'] != 0]
#print(perturbed[[col for col in df.columns if 'tau_fext' in col]].head(5))
#print(perturbed[[c for c in df.columns if 'tau_Jt' in c]].head(5))
#print(df.columns.tolist())
#print(df['block_id'].value_counts().sort_index())

# ── ADD THIS HERE — check trial lengths before filtering ─────────────────
#trial_lengths = df[df['trial_id'] > 0].groupby('trial_id').size()
#print("Trial length stats:")
#print(trial_lengths.describe())
#print("Shortest trial:", trial_lengths.min(), "frames")
#print("Longest trial:",  trial_lengths.max(), "frames")

CSV_OUT    = 'Data_1/Subject3_features.csv'

df_raw = pd.read_csv(CSV_OUT)
print(df[df['block_id'] == 0]['motor_force'].abs().describe())

print(df[df['block_id'] == 0]['time'].max())  # how long is block 0?
print(df['block_id'].value_counts().sort_index())  # how many frames per block?