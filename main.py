import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import pickle

# Python: 3.8.6
# scipy: 1.5.4
# numpy: 1.18.5
# matplotlib: 3.3.2
# pandas: 1.1.4
# sklearn: 0.23.2

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

def strip_player_name(name):
    return name.replace(' ', '').replace('-', '').replace('.', '')

for year in years:
    # Create dataframe with wr fantasy averages through first x weeks.
    for week in assessment_weeks:
        print('{} | Week {}'.format(year, week))
        wr_data_weekly = get_wr_df(week, year)

        if wr_assessment_df is None:
            wr_assessment_df = wr_data_weekly
        else:
            wr_assessment_df = pd.concat([wr_assessment_df, wr_data_weekly])

    wr_assessment_df = get_wr_aggregates(wr_assessment_df)

    # Create dataframe with wr fantasy averages during last x weeks.
    for week in judgement_weeks:
        print('{} | Week {}'.format(year, week))
        wr_data_weekly = get_wr_df(week, year)

        if wr_judgement_df is None:
            wr_judgement_df = wr_data_weekly
        else:
            wr_judgement_df = pd.concat([wr_judgement_df, wr_data_weekly])

    wr_judgement_df = get_wr_aggregates(wr_judgement_df)

    # Join tables for evaluation.
    wr_year_full_data = wr_assessment_df.assign(Judge_Std=wr_judgement_df.get('Standard'),
                                                Judge_Half=wr_judgement_df.get('HalfPPR'),
                                                Judge_PPR=wr_judgement_df.get('FullPPR'))
    wr_year_full_data = wr_year_full_data[wr_year_full_data.get('Judge_Std').notna()]
    wr_year_full_data = wr_year_full_data.assign(year=year*np.ones(wr_year_full_data.shape[0]))
    wr_year_full_data = wr_year_full_data.reset_index()
    wr_year_full_data = wr_year_full_data.assign(id=wr_year_full_data.get('Player').apply(strip_player_name) +
                                                    wr_year_full_data.get('year').apply(int).apply(str))
    wr_year_full_data = wr_year_full_data.set_index('id')
    if wr_full_data is None:
        wr_full_data = wr_year_full_data
    else:
        wr_full_data = pd.concat([wr_full_data, wr_year_full_data])

print(wr_full_data)
print(wr_full_data.columns)

X = np.array(wr_full_data.get(['Rec_pg', 'Tgt_pg', 'Yds_pg', 'Standard', 'HalfPPR', 'FullPPR']))
y = np.array(wr_full_data.get(['Judge_Std', 'Judge_Half', 'Judge_PPR']))
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

'''
# Find Good model.
best_acc = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best_acc:
        best_acc = acc
        with open('wr_fantasy_model.pickle', 'wb') as f:
            pickle.dump(linear, f)
print(best_acc)
'''

pickle_in = open('wr_fantasy_model.pickle', 'rb')
linear = pickle.load(pickle_in)
print(linear.score(x_test, y_test))

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Plots!
wr_full_data.plot(x='Rec_pg', y='Judge_Std', kind='scatter', alpha=0.5)
plt.title('Rec_pg vs Judge_Std', fontname='DejaVu Sans', fontsize=18)
wr_full_data.plot(x='Tgt_pg', y='Judge_Std', kind='scatter', alpha=0.5)
plt.title('Tgt_pg vs Judge_Std', fontname='DejaVu Sans', fontsize=18)
wr_full_data.plot(x='Yds_pg', y='Judge_Std', kind='scatter', alpha=0.5)
plt.title('Yds_pg vs Judge_Std', fontname='DejaVu Sans', fontsize=18)




#plt.show()



