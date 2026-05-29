import pandas as pd

# 1. Load data
# Try sep=None with engine='python' to let pandas auto-detect if it's a tab or comma
df = pd.read_csv('list_data.csv', sep=None, engine='python', on_bad_lines='skip')

# 2. Clean column names (Remove spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Debug: Print columns to verify 'province' exists
print("Detected columns:", df.columns.tolist())

if 'province' not in df.columns:
    print("Error: Could not find 'province' column. Check your file headers.")
else:
    # 3. Clean province name — remove number prefix
    df['province_clean'] = df['province'].astype(str).str.replace(r'^\d+\s+', '', regex=True)

    # 4. Calculate age group distribution per province
    age_dist = (
        df.groupby(['province_clean', 'age_group'])
        .size()
        .reset_index(name='count')
    )

    # 5. Convert to percentage within each province
    age_dist['percentage'] = (
        age_dist.groupby('province_clean')['count']
        .transform(lambda x: (x / x.sum() * 100).round(2))
    )

    # 6. Pivot for clean view
    age_table = age_dist.pivot(
        index='province_clean',
        columns='age_group',
        values='percentage'
    ).fillna(0)

    print(age_table)
    age_table.to_csv('province_age_distribution.csv')