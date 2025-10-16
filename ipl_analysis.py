"""
IPL Data Analysis - display plots instead of saving to files

How to use:
1) Install dependencies:
   pip install pandas matplotlib seaborn

2) Download dataset from Kaggle:
   kaggle datasets download -d manasgarg/ipl
   unzip ipl.zip -d data

3) Run:
   python ipl_analysis.py

This version will **display the plots one by one** instead of saving them to outputs/plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

DATA_DIR = 'data'
OUT_DIR = 'outputs'

os.makedirs(OUT_DIR, exist_ok=True)

# Load data
matches_path = os.path.join(DATA_DIR, 'matches.csv')
deliveries_path = os.path.join(DATA_DIR, 'deliveries.csv')

if not os.path.exists(matches_path) or not os.path.exists(deliveries_path):
    raise FileNotFoundError('Place matches.csv and deliveries.csv into the data/ folder.')

matches = pd.read_csv(matches_path)
deliveries = pd.read_csv(deliveries_path)

# 1) Top run-scorers (career)
runs_by_player = deliveries.groupby('batsman')['batsman_runs'].sum().reset_index()
runs_by_player = runs_by_player.sort_values('batsman_runs', ascending=False)
top_runs = runs_by_player.head(30)
top_runs.to_csv(os.path.join(OUT_DIR, 'top_scorers.csv'), index=False)

plt.figure(figsize=(10,6))
sns.barplot(data=top_runs.head(15), x='batsman_runs', y='batsman')
plt.title('Top 15 Run-scorers (Career)')
plt.xlabel('Total Runs')
plt.ylabel('Batsman')
plt.tight_layout()
plt.show()

# 2) Top wicket-takers (career)
wicket_types = ['bowled', 'caught', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
wickets = deliveries[deliveries['dismissal_kind'].isin(wicket_types)]
wickets_by_bowler = wickets.groupby('bowler').size().reset_index(name='wickets')
wickets_by_bowler = wickets_by_bowler.sort_values('wickets', ascending=False)
top_wkts = wickets_by_bowler.head(30)
top_wkts.to_csv(os.path.join(OUT_DIR, 'top_wicket_takers.csv'), index=False)

plt.figure(figsize=(10,6))
sns.barplot(data=top_wkts.head(15), x='wickets', y='bowler')
plt.title('Top 15 Wicket-takers (Career)')
plt.xlabel('Wickets')
plt.ylabel('Bowler')
plt.tight_layout()
plt.show()

# 3) Compare teams' win percentages by year
matches_clean = matches.copy()
matches_played = matches_clean[~matches_clean['result'].isin(['no result'])]

teams = pd.concat([matches_played[['season','team1']].rename(columns={'team1':'team'}),
                   matches_played[['team2','season']].rename(columns={'team2':'team','season':'season'})])

matches_played_count = teams.groupby(['season','team']).size().reset_index(name='matches_played')

wins = matches_played.groupby(['season','winner']).size().reset_index(name='wins')
wins = wins.rename(columns={'winner':'team'})

win_stats = pd.merge(matches_played_count, wins, on=['season','team'], how='left')
win_stats['wins'] = win_stats['wins'].fillna(0)
win_stats['win_pct'] = win_stats['wins'] / win_stats['matches_played'] * 100
win_stats = win_stats.sort_values(['season','win_pct'], ascending=[True, False])
win_stats.to_csv(os.path.join(OUT_DIR, 'win_pct_by_year.csv'), index=False)

for season, grp in win_stats.groupby('season'):
    top6 = grp.nlargest(6, 'win_pct')
    plt.figure(figsize=(8,5))
    sns.barplot(data=top6, x='win_pct', y='team')
    plt.title(f'Top 6 Teams by Win % - Season {season}')
    plt.xlabel('Win Percentage')
    plt.tight_layout()
    plt.show()

# 4) Visualize match outcomes and toss decisions
outcomes = matches_clean.groupby(['season','result']).size().reset_index(name='count')
plt.figure(figsize=(10,6))
for res in outcomes['result'].unique():
    subset = outcomes[outcomes['result']==res]
    plt.plot(subset['season'], subset['count'], marker='o', label=res)
plt.legend()
plt.title('Match Results by Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.tight_layout()
plt.show()

if 'toss_decision' in matches_clean.columns and 'toss_winner' in matches_clean.columns:
    toss_outcomes = matches_clean.groupby(['toss_decision'], group_keys=False).apply(
        lambda df: (df['toss_winner'] == df['winner']).mean()
    ).reset_index(name='pct_win_if_toss_win')
    toss_outcomes.to_csv(os.path.join(OUT_DIR, 'toss_outcomes.csv'), index=False)

    plt.figure(figsize=(10,6))
    sns.countplot(data=matches_clean, x='toss_decision', order=matches_clean['toss_decision'].unique())
    plt.title('Toss Decisions (overall)')
    plt.xlabel('Decision')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

summary = {
    'num_matches': [len(matches_clean)],
    'num_seasons': [matches_clean['season'].nunique()],
    'num_players': [deliveries['batsman'].nunique()+deliveries['bowler'].nunique()]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)

print('Done. Summary and CSV outputs saved to', OUT_DIR)
