import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # Красивые графики

# FILE_PATH = 'visual.csv'
FILE_PATH = 'visual.xls'

#для csv
# df = pd.read_csv(FILE_PATH, sep=';')
# df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', '2017', '2018', '2019'], axis=1, inplace=True)# удаляем для csv

# для xls
# для xls еще нужна бибилиотека xlrd
df = pd.read_excel(FILE_PATH, sheet_name='Data') # вкладки снизу экселя
df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', '2017', '2018', '2019'], axis=1, inplace=True)# удаляем для xls

years = np.zeros(1990-1960)

for inx in range(len(years)):
    years[inx] = int(1960 + inx)
    df.drop(columns=[str(int(years[inx]))], axis=1, inplace=True)

# Total sum
df.set_index('Country Name').sum().plot()

# top 5 countries' inflation
country_list = df['Country Name'].to_numpy()
combined = np.zeros(len(country_list))
for i in range(len(country_list)):
    sum_ = df.iloc[[i]].sum(level=0)
    sum_ = (sum_.sum(axis=1)).to_numpy()
    combined[i] = float(sum_/1000000)

df = df.assign(total=combined)
df = df.sort_values(by='total', ascending=False)
print(df)
top = df.head(5)
top = top.drop(columns=['total'], axis=1)
print("top", top)
top.set_index('Country Name').T.plot()

plt.show()
