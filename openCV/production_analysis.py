import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stat
import matplotlib.pyplot as plt

'''
DEFS
'''
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

#function to put ages in buckets
def bin_months(df):
    bins = pd.Series(['2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01',
            '2017-06-01', '2017-07-01', '2017-08-01', '2017-09-01', '2017-10-01',
            '2017-11-01', '2017-12-01'])

    print(pd.to_datetime(df['date_time']))
    print(pd.to_datetime(bins))
    group_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul',
                   'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    months = pd.cut(pd.to_datetime(df['date_time']), pd.to_datetime(bins), labels=group_names)
    df.month_packed = months
    return df



'''
'''

#read data
orig = pd.read_csv("production_2017.csv")
#strip time from date_time
orig['date_time'] = orig['date_time'].str[:10]
#fill NaN with zeros
orig['berry_grade'] = orig['berry_grade'].fillna(0)
#replace halcon tuple with 'Pass'
orig = orig[orig['berry_grade'] != 'HalconDotNet.HTupleElements']
# orig = orig.replace('HalconDotNet.HTupleElements', 'Overripe')

# orig.loc[(orig['status'] == 'Fail') & (orig['berry_grade'] == 0), 'berry_grade'] = 'Underripe'

print(orig)

total = orig.loc[orig['berry_grade'] > 0, 'punnet_count'].count()
#remove 'Passes'
orig = orig[orig['berry_grade'].str.contains('Pass') == False]

#only entries that have berry_grade
prod = orig[orig['berry_grade'] > 0]

#only after this date
partial = prod[prod['date_time'] > '2017-09-01']

# print(prod.sample())

dataset = prod

#count class totals
under = dataset[dataset['berry_grade'].str.contains('Underripe')].count()
over = dataset[dataset['berry_grade'].str.contains('Overripe')].count()
small = dataset[dataset['berry_grade'].str.contains('Too Small')].count()
foreign = dataset[dataset['berry_grade'].str.contains('Foreign')].count()
bruise = dataset[dataset['berry_grade'].str.contains('Bruise')].count()
fails = dataset[dataset['status'].str.contains('Fail')].count()

dataset['date_time'] = dataset['date_time'].str[:7]

groups_per_month = dataset.groupby(['date_time', 'berry_grade']).count()



'''
DISPLAY
'''
#plotting pie chart
labels = 'Under Ripe', 'Over Ripe'#, 'Too Small', 'Foreign Object', 'Bruise'
values = [under[0], over[0]]#, small[0], foreign[0], bruise[0]]
fails = sum(values)
explode = (0, 0)#, 0.2, 0.2, 0.2)  # only "explode" the foreign and bruise slices
fig1, ax1 = plt.subplots()
ax1.pie(values, explode=explode, labels=labels, autopct=make_autopct(values),
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total - ' + str(total) + '   Failed - ' + str(fails))



groups = groups_per_month.sort_index()

groups.drop(['id', 'status', 'image_path', 'berry_count', 'reject_count', 'min_distance', 'max_distance'], axis=1, inplace=True)

df = pd.DataFrame(data=groups)

df['punnet_count'] = df['punnet_count'].fillna(0)


df.unstack(level=0).T.plot(kind='bar', subplots=False)


# Show all plots
plt.show()

#save to csv
# prod_sum.to_csv('prod_summary_2017.csv', sep='\t', encoding='utf-8')
