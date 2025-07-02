import matplotlib.pyplot as plt
import pandas as pd

# Define the raw totals from earlier analysis
data = {
    'Age Group': ['Children', 'Young Adult', 'Adult', 'Senior'],
    '1900': [33800, 19650, 18050, 4950],
    '2000': [80150, 58500, 96900, 45700]
}

df = pd.DataFrame(data)

# ----- Chart 1: Stacked Bar Chart -----
fig1, ax1 = plt.subplots(figsize=(8, 5))
bar_width = 0.35
x = range(len(df['Age Group']))

ax1.bar(x, df['1900'], width=bar_width, label='1900')
ax1.bar([i + bar_width for i in x], df['2000'], width=bar_width, label='2000')

ax1.set_xticks([i + bar_width / 2 for i in x])
ax1.set_xticklabels(df['Age Group'])
ax1.set_ylabel('Population (Thousands)')
ax1.set_title('Age Group Population Comparison (1900 vs 2000)')
ax1.legend()

plt.tight_layout()

# ----- Chart 2: Line Chart (Percent Composition Over Time) -----
# Normalize for percent composition
df_percent = df.copy()
df_percent['1900'] = df_percent['1900'] / df_percent['1900'].sum() * 100
df_percent['2000'] = df_percent['2000'] / df_percent['2000'].sum() * 100

fig2, ax2 = plt.subplots(figsize=(8, 5))
for idx, row in df_percent.iterrows():
    ax2.plot(['1900', '2000'], [row['1900'], row['2000']], marker='o', label=row['Age Group'])

ax2.set_ylabel('Percent of Total Population')
ax2.set_title('Age Group Composition Shift (1900 to 2000)')
ax2.legend()

plt.tight_layout()
plt.show()

