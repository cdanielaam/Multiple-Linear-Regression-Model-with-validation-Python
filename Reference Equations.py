#!/usr/bin/env python
# coding: utf-8

#In[1]
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import seaborn as sns
import scipy as sp
from scipy import stats

#Import data:
SEL2020 = pd.read_csv("/Users/pixel/Desktop/Valores referencia espirometria/Equacoes 2020/SEL2020.csv")

#Preview data:
print(SEL2020.head(5))

#Subset selection:
ARIA = SEL2020[SEL2020["Project"] == 0]
ARIA_Boys = ARIA[ARIA["Gender"] == "Male"]
ARIA_Girls = ARIA[ARIA["Gender"] == "Female"]
G21 = SEL2020[SEL2020["Project"] == 1] 
G21_Boys = G21[G21['Gender'] == 'Male']
G21_Girls = G21[G21['Gender'] == 'Female']

#In[]
#Descriptive analysis for Derivation Cohort ARIA:
# n and % for Gender ARIA:
type("Gender")
ARIA['Gender'].value_counts()

#Males:
(267/481)*100

#Females:

(214/481)*100

counts = [267, 214]
percentage = [(267/481)*100, (214/481)*100]
bars = ('Male', 'Female')

y_pos = np.arange(len(bars))
plt.bar(y_pos, counts, color = ['blue', 'green'])
plt.xticks(y_pos, bars)
plt.show()

#In[]
y_pos = np.arange(len(bars))
plt.bar(y_pos, percentage, color = ['blue', 'green'])
plt.xticks(y_pos, bars)
plt.show()

#In[]
#Age range:
ARIA.info()
Mean_ARIA_Age = ARIA['Idade'].mean()
print(Mean_ARIA_Age)
SD_ARIA_Age = ARIA['Idade'].std()
print(SD_ARIA_Age)
Min_ARIA_Age = ARIA['Idade'].min()
print(Min_ARIA_Age)
Max_ARIA_Age = ARIA['Idade'].max()
print(Max_ARIA_Age)
#In[]
#Age comparisson between Boys and Girls:
sns.boxplot( x=ARIA["Gender"], y=ARIA["Idade"] )
# In[]
sns.distplot( ARIA_Boys["Idade"] , color="skyblue", label="Male Age")
sns.distplot( ARIA_Girls["Idade"] , color="red", label="Female Age")
# In[]
stats.ttest_ind(ARIA_Boys["Idade"],ARIA_Girls["Idade"], equal_var = True)
#In[]
Mean_ARIA_Boys_Age = ARIA_Boys['Idade'].mean()
print(Mean_ARIA_Boys_Age)
SD_ARIA_Boys_Age = ARIA_Boys['Idade'].std()
print(SD_ARIA_Boys_Age)
Min_ARIA_Boys_Age = ARIA_Boys['Idade'].min()
print(Min_ARIA_Boys_Age)
Max_ARIA_Boys_Age = ARIA_Boys['Idade'].max()
print(Max_ARIA_Boys_Age)
# In[]
Mean_ARIA_Girls_Age = ARIA_Girls['Idade'].mean()
print(Mean_ARIA_Girls_Age)
SD_ARIA_Girls_Age = ARIA_Girls['Idade'].std()
print(SD_ARIA_Girls_Age)
Min_ARIA_Girls_Age = ARIA_Girls['Idade'].min()
print(Min_ARIA_Girls_Age)
Max_ARIA_Girls_Age = ARIA_Girls['Idade'].max()
print(Max_ARIA_Girls_Age)
# %%
