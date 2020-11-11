import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# What best projects a receiver's fantasy output?
# Targets
# Catches
# Yards
# TDs
# Ft Points
# Some other combo?

mpl.rcParams['figure.figsize'] = (8, 5)
plt.style.use('ggplot')

years = np.arange(1999, 2020, 1)
assessment_weeks = np.arange(1, 7, 1)
judgement_weeks = np.arange(7, 18, 1)

YEAR_TEMP = 2019
wr_assessment_df = None
wr_judgement_df = None
wr_full_data = None

def get_wr_df(week, year):
    """Returns a dataframe with relevant fantasy stats for a particular week"""
    player_data = pd.read_csv('weekly/{}/week{}.csv'.format(year, week))
    wr_data_weekly = (
        player_data[player_data.get('Pos') == 'WR']
        .get(['Player', 'Pos', 'Rec', 'Tgt', 'ReceivingYds',
              'StandardFantasyPoints', 'HalfPPRFantasyPoints', 'PPRFantasyPoints'])
    )
    wr_data_weekly = wr_data_weekly.assign(gp=np.ones(wr_data_weekly.shape[0]))  # Add Games Played

    return wr_data_weekly

def get_wr_aggregates(wr_df):
    """When given a df of weekly boxscores, returns a df of average stats"""
    wr_data_grouped = wr_df.groupby(by='Player').sum()
    wr_data_aggregates = wr_data_grouped.assign(Rec_pg=wr_data_grouped.get('Rec') / wr_data_grouped.get('gp'),
                                                Tgt_pg=wr_data_grouped.get('Tgt') / wr_data_grouped.get('gp'),
                                                Yds_pg=wr_data_grouped.get('ReceivingYds') / wr_data_grouped.get('gp'),
                                                Standard=wr_data_grouped.get(
                                                    'StandardFantasyPoints') / wr_data_grouped.get('gp'),
                                                HalfPPR=wr_data_grouped.get(
                                                    'HalfPPRFantasyPoints') / wr_data_grouped.get('gp'),
                                                FullPPR=wr_data_grouped.get('PPRFantasyPoints') / wr_data_grouped.get(
                                                    'gp'))

    wr_data_aggregates = wr_data_aggregates.get(['gp', 'Rec_pg', 'Tgt_pg', 'Yds_pg', 'Standard', 'HalfPPR', 'FullPPR'])
    wr_data_aggregates = wr_data_aggregates[wr_data_aggregates.get('gp') >= 5]
    return wr_data_aggregates

# Create dataframe with wr fantasy averages through first x weeks.
for week in assessment_weeks:
    wr_data_weekly = get_wr_df(week, YEAR_TEMP)

    if wr_assessment_df is None:
        wr_assessment_df = wr_data_weekly
    else:
        wr_assessment_df = pd.concat([wr_assessment_df, wr_data_weekly])

wr_assessment_df = get_wr_aggregates(wr_assessment_df)

# Create dataframe with wr fantasy averages during last x weeks.
for week in judgement_weeks:
    wr_data_weekly = get_wr_df(week, YEAR_TEMP)

    if wr_judgement_df is None:
        wr_judgement_df = wr_data_weekly
    else:
        wr_judgement_df = pd.concat([wr_judgement_df, wr_data_weekly])

wr_judgement_df = get_wr_aggregates(wr_judgement_df)

# Join tables for evaluation.
wr_full_data = wr_assessment_df.assign(Judge_Std=wr_judgement_df.get('Standard'),
                                       Judge_Half=wr_judgement_df.get('HalfPPR'),
                                       Judge_PPR=wr_judgement_df.get('FullPPR'))
wr_full_data = wr_full_data[wr_full_data.get('Judge_Std').notna()]

print(wr_full_data)
print(wr_full_data.columns)

# Plots!
wr_full_data.plot(x='Rec_pg', y='Judge_Std', kind='scatter')

plt.show()



