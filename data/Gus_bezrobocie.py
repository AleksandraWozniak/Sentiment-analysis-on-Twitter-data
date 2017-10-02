
# coding: utf-8

from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

# Importing data from the Polish Statistical Office website (GUS)
# (Pobranie danych ze strony GUS)
t = Table.read('http://stat.gov.pl/obszary-tematyczne/rynek-pracy/bezrobocie-rejestrowane/stopa-bezrobocia-w-latach-1990-2017,4,1.html', format='html', 
               fill_values=None)

# writing original data into csv file
t.write('Guso.csv', format='csv')

#t.show_in_browser()  

#df = t.to_pandas()

df = pd.read_csv('Guso.csv')

df

# Data needs cleaning and preprocessing (for year 2002 there are e.g. 2 rows with values)
# (dane trzeba oczyscic i przeksztalcic)
# (dla roku 2002 w tabeli mamy po dwie wartości, trzeba jedną usunac)
df.iloc[15] = df.iloc[15].apply(lambda x: str(x)[:4])

df

# Sign "*" needs to be excluded, insert proper decimals 
# (trzeba wyeliminować gwiazdki przy liczbach, a)
# (miejsca dziesiętne musza byc oddzielone kropką, a nie przecinkiem w celu wizualizacji danych)

for i in range(len(df)):
    df.iloc[i] = df.iloc[i].apply(lambda x: str(x).replace('*',''))
    df.iloc[i,1:] = df.iloc[i,1:].apply(lambda x: float(x.replace(',','.')))

df

df.dtypes

df['luty'] = df['luty'].astype(float)
df.dtypes

# Changing values into float type
# (zamiana wartosci w kolumnach na float)
df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]] = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].astype(float)
#df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]] = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].apply(pd.to_numeric, errors='coerce')
#df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]] = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].convert_objects(convert_numeric=True)

df.dtypes

# Changing first column into date format
# (zamiana pierwszej kolumny na format daty)
df.iloc[:,[0]] = df.iloc[:,[0]].apply(pd.to_datetime, errors='coerce')

df.dtypes 


# Calculating dynamics change
# (wyliczenie dynamiki zmian)
df['R/r'] = np.nan
for i in range(len(df)-1):
    df['R/r'][i] = round((df.iloc[i,1]/df.iloc[i+1,1]-1)*100,)

# Writing processed data into new csv file
df.to_csv('Gus_zmiana.csv', index=False)


# Changing data into time series

#(dane zamieniam na szereg czasowy)

s = pd.Series()
for i in range(len(df)):
    temp = pd.Series()
    temp = df.iloc[-i-1,1:-1] # luty 2017 8.5
    temp.index = str(df.iloc[-i-1,0])[:4]+' '+df.iloc[-i-1,1:-1].index # 1990-styczen 1990-luty itd. 
    s = s.append(temp)

s = s.reset_index(drop=False)
s = s.rename(columns = {0:'stopa'})
#s
s = s.loc[0:331]
s

# Creating time series from the data
# (tworzenie szeregu czasowego)

s['stopa'] = s['stopa'].apply(pd.to_numeric, errors='coerce')

t = pd.date_range('1990-01-01', periods=len(s), freq='m')

t = pd.DataFrame(t)

s = pd.merge(s, t, left_index=True, right_index=True)

del(s['index'])

s = s.rename(columns = {0:'data'})

s

#s.drop(s.index[0])

# Date as index
# (data jako indeks)

s.index = s['data'] 
del(s['data']) # del df.index.name
s.head()

s.index

pd.DataFrame(s)


# Plotting unemployment rate
# (Analiza wykresu moze pomoc w zidentyfikowaniu regularnych wzorców wystepujacych w danych, takich sezonowosc)
dates = pd.date_range('1990-01', '2017-08', freq='M')
labels= dates.strftime('%y %b')
plt.figure(figsize=(12,6))
plt.plot(s.values, lw=5, color='b')

plt.title('Stopa bezrobocia w Polsce w latach 1990-2017 po miesiącach')
plt.xlabel('Miesiące')
plt.ylabel('Stopa bezrobocia w %')
plt.show()

# Saving data to file
s.to_csv('Gus_bezrobocie.csv')

