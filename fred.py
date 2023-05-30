import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from full_fred.fred import Fred

fred = Fred("C:/Users/kanld/Desktop/api_key.txt")

def clean(df):
    df = df.drop(columns = ['realtime_start', 'realtime_end'])
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df['value'].replace('.', method = 'bfill') # formatted in standard calendar day and keeps holiday values as '.'
    df = df.apply(float)

    return df

''' id, limit, offset, observation_start, observation_end'''

df_sp500 = fred.get_series_df('SP500').pipe(clean)
df_ust10y = fred.get_series_df('T10Y2Y').pipe(clean)


sns.lineplot(data = df_sp500, x = df_sp500.index, y = df_sp500.values)
sns.lineplot(data = df_ust10y, x= df_ust10y.index, y = df_ust10y.values)


plt.show()