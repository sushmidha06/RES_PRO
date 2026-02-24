import pandas as pd
df = pd.read_csv('dataset/resume_dataset.csv', engine='python')
label_col = [col for col in df.columns if 'job_position_name' in col][0]
for i in range(2):
    print(f"Row {i}")
    print(f"Target: {df[label_col].iloc[i]}")
    print(f"Objective: {df['career_objective'].iloc[i]}")
    print(f"Skills: {df['skills'].iloc[i]}")
    print("-" * 50)
