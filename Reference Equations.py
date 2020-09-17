#!/usr/bin/env python
# coding: utf-8

#In[1]
import pandas as pd
#%%
import os
#%%
import matplotlib.pyplot as plt
#%%
from matplotlib.ticker import StrMethodFormatter
#%%
import numpy as np
#%%
import seaborn as sns
#%%
import scipy as scipy
#%%
from scipy import stats
#%%
import sklearn
#%%
from sklearn import datasets, linear_model
#%%
import sklearn.metrics 
#%%
from statsmodels.formula.api import ols
#%%
from statsmodels.stats.anova import anova_lm
#%%
import statsmodels.api as sm
#%%
#Import data:
SEL2020 = pd.read_csv("/Users/pixel/Desktop/Valores referencia espirometria/Equacoes 2020/SEL2020.csv")

#Preview data:
print(SEL2020.head(5))
#%%
#Subset selection:
ARIA = SEL2020[SEL2020["Project"] == 0]
#%%
ARIA_Boys = ARIA[ARIA["Gender"] == "Male"]
#%%
ARIA_Girls = ARIA[ARIA["Gender"] == "Female"]
#%%
G21 = SEL2020[SEL2020["Project"] == 1] 
#%%
G21_Boys = G21[G21['Gender'] == 'Male']
#%%
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
plt.bar(y_pos, counts, color = ['skyblue', 'orange'])
plt.xticks(y_pos, bars)
plt.show()

#In[]
y_pos = np.arange(len(bars))
plt.bar(y_pos, percentage, color = ['skyblue', 'orange'])
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
# In[]
#Height comparisson Boys and Girls ARIA:
stats.ttest_ind(ARIA_Boys["Altura_cm"],ARIA_Girls["Altura_cm"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["Altura_cm"] , color="skyblue", label="Boys Height")
sns.distplot( ARIA_Girls["Altura_cm"] , color="red", label="Girls Height")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["Altura_cm"] )
# %%
Mean_ARIA_Boys_H = ARIA_Boys['Altura_cm'].mean()
print(Mean_ARIA_Boys_H)
SD_ARIA_Boys_H = ARIA_Boys['Altura_cm'].std()
print(SD_ARIA_Boys_H)
Min_ARIA_Boys_H = ARIA_Boys['Altura_cm'].min()
print(Min_ARIA_Boys_H)
Max_ARIA_Boys_H = ARIA_Boys['Altura_cm'].max()
print(Max_ARIA_Boys_H)
# %%
Mean_ARIA_Girls_H = ARIA_Girls['Altura_cm'].mean()
print(Mean_ARIA_Girls_H)
SD_ARIA_Girls_H = ARIA_Girls['Altura_cm'].std()
print(SD_ARIA_Girls_H)
Min_ARIA_Girls_H = ARIA_Girls['Altura_cm'].min()
print(Min_ARIA_Girls_H)
Max_ARIA_Girls_H = ARIA_Girls['Altura_cm'].max()
print(Max_ARIA_Girls_H)
# %%
#Weight comparisson Boys and Girls ARIA:
stats.ttest_ind(ARIA_Boys["Peso"],ARIA_Girls["Peso"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["Peso"] , color="skyblue", label="Boys Weight")
sns.distplot( ARIA_Girls["Peso"] , color="red", label="Girls Weight")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["Peso"] )
# %%
Mean_ARIA_Boys_W = ARIA_Boys['Peso'].mean()
print(Mean_ARIA_Boys_W)
SD_ARIA_Boys_W = ARIA_Boys['Peso'].std()
print(SD_ARIA_Boys_W)
Min_ARIA_Boys_W = ARIA_Boys['Peso'].min()
print(Min_ARIA_Boys_W)
Max_ARIA_Boys_W = ARIA_Boys['Peso'].max()
print(Max_ARIA_Boys_W)
# %%
Mean_ARIA_Girls_W = ARIA_Girls['Peso'].mean()
print(Mean_ARIA_Girls_W)
SD_ARIA_Girls_W = ARIA_Girls['Peso'].std()
print(SD_ARIA_Girls_W)
Min_ARIA_Girls_W = ARIA_Girls['Peso'].min()
print(Min_ARIA_Girls_W)
Max_ARIA_Girls_W = ARIA_Girls['Peso'].max()
print(Max_ARIA_Girls_W)
# %%
#BMI comparisson Boys and Girls ARIA:
stats.ttest_ind(ARIA_Boys["BMI"],ARIA_Girls["BMI"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["BMI"] , color="skyblue", label="Boys BMI")
sns.distplot( ARIA_Girls["BMI"] , color="red", label="Girls BMI")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["BMI"] )
# %%
Mean_ARIA_Boys_BMI = ARIA_Boys['BMI'].mean()
print(Mean_ARIA_Boys_BMI)
SD_ARIA_Boys_BMI = ARIA_Boys['BMI'].std()
print(SD_ARIA_Boys_BMI)
Min_ARIA_Boys_BMI = ARIA_Boys['BMI'].min()
print(Min_ARIA_Boys_BMI)
Max_ARIA_Boys_BMI = ARIA_Boys['BMI'].max()
print(Max_ARIA_Boys_BMI)
# %%
Mean_ARIA_Girls_BMI = ARIA_Girls['BMI'].mean()
print(Mean_ARIA_Girls_BMI)
SD_ARIA_Girls_BMI = ARIA_Girls['BMI'].std()
print(SD_ARIA_Girls_BMI)
Min_ARIA_Girls_BMI = ARIA_Girls['BMI'].min()
print(Min_ARIA_Girls_BMI)
Max_ARIA_Girls_BMI = ARIA_Girls['BMI'].max()
print(Max_ARIA_Girls_BMI)
# %%
#Comparisson two-sample t-test spirometric parameters between Male and Female participants in ARIA:
#FVC_pre:
stats.ttest_ind(ARIA_Boys["FVC_pre"],ARIA_Girls["FVC_pre"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["FVC_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( ARIA_Girls["FVC_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["FVC_pre"] )
# %%
Mean_ARIA_Boys_FVC = ARIA_Boys['FVC_pre'].mean()
print(Mean_ARIA_Boys_FVC)
SD_ARIA_Boys_FVC = ARIA_Boys['FVC_pre'].std()
print(SD_ARIA_Boys_FVC)
Min_ARIA_Boys_FVC = ARIA_Boys['FVC_pre'].min()
print(Min_ARIA_Boys_FVC)
Max_ARIA_Boys_FVC = ARIA_Boys['FVC_pre'].max()
print(Max_ARIA_Boys_FVC)
# %%
Mean_ARIA_Girls_FVC = ARIA_Girls['FVC_pre'].mean()
print(Mean_ARIA_Girls_FVC)
SD_ARIA_Girls_FVC = ARIA_Girls['FVC_pre'].std()
print(SD_ARIA_Girls_FVC)
Min_ARIA_Girls_FVC = ARIA_Girls['FVC_pre'].min()
print(Min_ARIA_Girls_FVC)
Max_ARIA_Girls_FVC = ARIA_Girls['FVC_pre'].max()
print(Max_ARIA_Girls_FVC)
# %%
#FEV1_pre:
stats.ttest_ind(ARIA_Boys["FEV1_pre"],ARIA_Girls["FEV1_pre"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["FEV1_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( ARIA_Girls["FEV1_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["FEV1_pre"] )
# %%
Mean_ARIA_Boys_FEV1 = ARIA_Boys['FEV1_pre'].mean()
print(Mean_ARIA_Boys_FEV1)
SD_ARIA_Boys_FEV1 = ARIA_Boys['FEV1_pre'].std()
print(SD_ARIA_Boys_FEV1)
Min_ARIA_Boys_FEV1 = ARIA_Boys['FEV1_pre'].min()
print(Min_ARIA_Boys_FEV1)
Max_ARIA_Boys_FEV1 = ARIA_Boys['FEV1_pre'].max()
print(Max_ARIA_Boys_FEV1)
# %%
Mean_ARIA_Girls_FEV1 = ARIA_Girls['FEV1_pre'].mean()
print(Mean_ARIA_Girls_FEV1)
SD_ARIA_Girls_FEV1 = ARIA_Girls['FEV1_pre'].std()
print(SD_ARIA_Girls_FEV1)
Min_ARIA_Girls_FEV1 = ARIA_Girls['FEV1_pre'].min()
print(Min_ARIA_Girls_FEV1)
Max_ARIA_Girls_FEV1 = ARIA_Girls['FEV1_pre'].max()
print(Max_ARIA_Girls_FEV1)
# %%
#FEF2575_pre:
stats.ttest_ind(ARIA_Boys["FEF2575_pre"],ARIA_Girls["FEF2575_pre"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["FEF2575_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( ARIA_Girls["FEF2575_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["FEF2575_pre"] )
# %%
Mean_ARIA_Boys_FEF2575 = ARIA_Boys['FEF2575_pre'].mean()
print(Mean_ARIA_Boys_FEF2575)
SD_ARIA_Boys_FEF2575 = ARIA_Boys['FEF2575_pre'].std()
print(SD_ARIA_Boys_FEF2575)
Min_ARIA_Boys_FEF2575 = ARIA_Boys['FEF2575_pre'].min()
print(Min_ARIA_Boys_FEF2575)
Max_ARIA_Boys_FEF2575 = ARIA_Boys['FEF2575_pre'].max()
print(Max_ARIA_Boys_FEF2575)
# %%
Mean_ARIA_Girls_FEF2575 = ARIA_Girls['FEF2575_pre'].mean()
print(Mean_ARIA_Girls_FEF2575)
SD_ARIA_Girls_FEF2575 = ARIA_Girls['FEF2575_pre'].std()
print(SD_ARIA_Girls_FEF2575)
Min_ARIA_Girls_FEF2575 = ARIA_Girls['FEF2575_pre'].min()
print(Min_ARIA_Girls_FEF2575)
Max_ARIA_Girls_FEF2575 = ARIA_Girls['FEF2575_pre'].max()
print(Max_ARIA_Girls_FEF2575)
# %%
#FEV1FVC_pre:
stats.ttest_ind(ARIA_Boys["FEV1FVC_pre"],ARIA_Girls["FEV1FVC_pre"], equal_var = True)
#%%
sns.distplot( ARIA_Boys["FEV1FVC_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( ARIA_Girls["FEV1FVC_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=ARIA["Gender"], y=ARIA["FEV1FVC_pre"] )
# %%
Mean_ARIA_Boys_FEV1FVC = ARIA_Boys['FEV1FVC_pre'].mean()
print(Mean_ARIA_Boys_FEV1FVC)
SD_ARIA_Boys_FEV1FVC = ARIA_Boys['FEV1FVC_pre'].std()
print(SD_ARIA_Boys_FEV1FVC)
Min_ARIA_Boys_FEV1FVC = ARIA_Boys['FEV1FVC_pre'].min()
print(Min_ARIA_Boys_FEV1FVC)
Max_ARIA_Boys_FEV1FVC = ARIA_Boys['FEV1FVC_pre'].max()
print(Max_ARIA_Boys_FEV1FVC)
# %%
Mean_ARIA_Girls_FEV1FVC = ARIA_Girls['FEV1FVC_pre'].mean()
print(Mean_ARIA_Girls_FEV1FVC)
SD_ARIA_Girls_FEV1FVC = ARIA_Girls['FEV1FVC_pre'].std()
print(SD_ARIA_Girls_FEV1FVC)
Min_ARIA_Girls_FEV1FVC = ARIA_Girls['FEV1FVC_pre'].min()
print(Min_ARIA_Girls_FEV1FVC)
Max_ARIA_Girls_FEV1FVC = ARIA_Girls['FEV1FVC_pre'].max()
print(Max_ARIA_Girls_FEV1FVC)
# %%
#Descriptive analysis for Validation Cohort G21:
# n and % for Gender G21:
type("Gender")
G21['Gender'].value_counts()
1538+1448
#%%
#Males:
(1538/2986)*100
#%%
#Females:

#%%
counts = [1538, 1448]
percentage = [(1538/2986)*100, (1448/2986)*100]
bars = ('Male', 'Female')

y_pos = np.arange(len(bars))
plt.bar(y_pos, counts, color = ['skyblue', 'orange'])
plt.xticks(y_pos, bars)
plt.show()

#In[]
y_pos = np.arange(len(bars))
plt.bar(y_pos, percentage, color = ['skyblue', 'orange'])
plt.xticks(y_pos, bars)
plt.show()
# %%
#Age range:
G21.info()
Mean_G21_Age = G21['Idade'].mean()
print(Mean_G21_Age)
SD_G21_Age = G21['Idade'].std()
print(SD_G21_Age)
Min_G21_Age = G21['Idade'].min()
print(Min_G21_Age)
Max_G21_Age = G21['Idade'].max()
print(Max_G21_Age)
#%%
#Gender comparissons:
#Age:
sns.boxplot( x=G21["Gender"], y=G21["Idade"] )
# In[]
sns.distplot( G21_Boys["Idade"] , color="skyblue", label="Male Age")
sns.distplot( G21_Girls["Idade"] , color="red", label="Female Age")
# In[]
stats.ttest_ind(G21_Boys["Idade"],G21_Girls["Idade"], equal_var = True)
#In[]
Mean_G21_Boys_Age = G21_Boys['Idade'].mean()
print(Mean_G21_Boys_Age)
SD_G21_Boys_Age = G21_Boys['Idade'].std()
print(SD_G21_Boys_Age)
Min_G21_Boys_Age = G21_Boys['Idade'].min()
print(Min_G21_Boys_Age)
Max_G21_Boys_Age = G21_Boys['Idade'].max()
print(Max_G21_Boys_Age)
# In[]
Mean_G21_Girls_Age = G21_Girls['Idade'].mean()
print(Mean_G21_Girls_Age)
SD_G21_Girls_Age = G21_Girls['Idade'].std()
print(SD_G21_Girls_Age)
Min_G21_Girls_Age = G21_Girls['Idade'].min()
print(Min_G21_Girls_Age)
Max_G21_Girls_Age = G21_Girls['Idade'].max()
print(Max_G21_Girls_Age)
# In[]
#Height comparisson Boys and Girls G21:
stats.ttest_ind(G21_Boys["Altura_cm"],G21_Girls["Altura_cm"], equal_var = True)
#%%
sns.distplot( G21_Boys["Altura_cm"] , color="skyblue", label="Boys Height")
sns.distplot( G21_Girls["Altura_cm"] , color="red", label="Girls Height")
#%%
sns.boxplot( x=G21["Gender"], y=G21["Altura_cm"] )
# %%
Mean_G21_Boys_H = G21_Boys['Altura_cm'].mean()
print(Mean_G21_Boys_H)
SD_G21_Boys_H = G21_Boys['Altura_cm'].std()
print(SD_G21_Boys_H)
Min_G21_Boys_H = G21_Boys['Altura_cm'].min()
print(Min_G21_Boys_H)
Max_G21_Boys_H = G21_Boys['Altura_cm'].max()
print(Max_G21_Boys_H)
# %%
Mean_G21_Girls_H = G21_Girls['Altura_cm'].mean()
print(Mean_G21_Girls_H)
SD_G21_Girls_H = G21_Girls['Altura_cm'].std()
print(SD_G21_Girls_H)
Min_G21_Girls_H = G21_Girls['Altura_cm'].min()
print(Min_G21_Girls_H)
Max_G21_Girls_H = G21_Girls['Altura_cm'].max()
print(Max_G21_Girls_H)
# %%
#Weight comparisson Boys and Girls G21:
stats.ttest_ind(G21_Boys["Peso"],G21_Girls["Peso"], equal_var = True)
#%%
sns.distplot( G21_Boys["Peso"] , color="skyblue", label="Boys Weight")
sns.distplot( G21_Girls["Peso"] , color="red", label="Girls Weight")
#%%
sns.boxplot( x=G21["Gender"], y=G21["Peso"] )
# %%
Mean_G21_Boys_W = G21_Boys['Peso'].mean()
print(Mean_G21_Boys_W)
SD_G21_Boys_W = G21_Boys['Peso'].std()
print(SD_G21_Boys_W)
Min_G21_Boys_W = G21_Boys['Peso'].min()
print(Min_G21_Boys_W)
Max_G21_Boys_W = G21_Boys['Peso'].max()
print(Max_G21_Boys_W)
# %%
Mean_G21_Girls_W = G21_Girls['Peso'].mean()
print(Mean_G21_Girls_W)
SD_G21_Girls_W = G21_Girls['Peso'].std()
print(SD_G21_Girls_W)
Min_G21_Girls_W = G21_Girls['Peso'].min()
print(Min_G21_Girls_W)
Max_G21_Girls_W = G21_Girls['Peso'].max()
print(Max_G21_Girls_W)
# %%
#BMI comparisson Boys and Girls G21:
stats.ttest_ind(G21_Boys["BMI"],G21_Girls["BMI"], equal_var = True)
#%%
sns.distplot( G21_Boys["BMI"] , color="skyblue", label="Boys BMI")
sns.distplot( G21_Girls["BMI"] , color="red", label="Girls BMI")
#%%
sns.boxplot( x=G21["Gender"], y=G21["BMI"] )
# %%
Mean_G21_Boys_BMI = G21_Boys['BMI'].mean()
print(Mean_G21_Boys_BMI)
SD_G21_Boys_BMI = G21_Boys['BMI'].std()
print(SD_G21_Boys_BMI)
Min_G21_Boys_BMI = G21_Boys['BMI'].min()
print(Min_G21_Boys_BMI)
Max_G21_Boys_BMI = G21_Boys['BMI'].max()
print(Max_G21_Boys_BMI)
# %%
Mean_G21_Girls_BMI = G21_Girls['BMI'].mean()
print(Mean_G21_Girls_BMI)
SD_G21_Girls_BMI = G21_Girls['BMI'].std()
print(SD_G21_Girls_BMI)
Min_G21_Girls_BMI = G21_Girls['BMI'].min()
print(Min_G21_Girls_BMI)
Max_G21_Girls_BMI = G21_Girls['BMI'].max()
print(Max_G21_Girls_BMI)
# %%
#Comparisson two-sample t-test spirometric parameters between Male and Female participants in G21:
#FVC_pre:
stats.ttest_ind(G21_Boys["FVC_pre"],G21_Girls["FVC_pre"], equal_var = True)
#%%
sns.distplot( G21_Boys["FVC_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( G21_Girls["FVC_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=G21["Gender"], y=G21["FVC_pre"] )
# %%
Mean_G21_Boys_FVC = G21_Boys['FVC_pre'].mean()
print(Mean_G21_Boys_FVC)
SD_G21_Boys_FVC = G21_Boys['FVC_pre'].std()
print(SD_G21_Boys_FVC)
Min_G21_Boys_FVC = G21_Boys['FVC_pre'].min()
print(Min_G21_Boys_FVC)
Max_G21_Boys_FVC = G21_Boys['FVC_pre'].max()
print(Max_G21_Boys_FVC)
# %%
Mean_G21_Girls_FVC =G21_Girls['FVC_pre'].mean()
print(Mean_G21_Girls_FVC)
SD_G21_Girls_FVC = G21_Girls['FVC_pre'].std()
print(SD_G21_Girls_FVC)
Min_G21_Girls_FVC = G21_Girls['FVC_pre'].min()
print(Min_G21_Girls_FVC)
Max_G21_Girls_FVC = G21_Girls['FVC_pre'].max()
print(Max_G21_Girls_FVC)
# %%
#FEV1_pre:
stats.ttest_ind(G21_Boys["FEV1_pre"],G21_Girls["FEV1_pre"], equal_var = True)
#%%
sns.distplot( G21_Boys["FEV1_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( G21_Girls["FEV1_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=G21["Gender"], y=G21["FEV1_pre"] )
# %%
Mean_G21_Boys_FEV1 = G21_Boys['FEV1_pre'].mean()
print(Mean_G21_Boys_FEV1)
SD_G21_Boys_FEV1 = G21_Boys['FEV1_pre'].std()
print(SD_G21_Boys_FEV1)
Min_G21_Boys_FEV1 = G21_Boys['FEV1_pre'].min()
print(Min_G21_Boys_FEV1)
Max_G21_Boys_FEV1 = G21_Boys['FEV1_pre'].max()
print(Max_G21_Boys_FEV1)
# %%
Mean_G21_Girls_FEV1 = G21_Girls['FEV1_pre'].mean()
print(Mean_G21_Girls_FEV1)
SD_G21_Girls_FEV1 = G21_Girls['FEV1_pre'].std()
print(SD_G21_Girls_FEV1)
Min_G21_Girls_FEV1 = G21_Girls['FEV1_pre'].min()
print(Min_G21_Girls_FEV1)
Max_G21_Girls_FEV1 = G21_Girls['FEV1_pre'].max()
print(Max_G21_Girls_FEV1)
# %%
#FEF2575_pre:
stats.ttest_ind(G21_Boys["FEF2575_pre"],G21_Girls["FEF2575_pre"], equal_var = True)
#%%
sns.distplot( G21_Boys["FEF2575_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( G21_Girls["FEF2575_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=G21["Gender"], y=G21["FEF2575_pre"] )
# %%
Mean_G21_Boys_FEF2575 = G21_Boys['FEF2575_pre'].mean()
print(Mean_G21_Boys_FEF2575)
SD_G21_Boys_FEF2575 = G21_Boys['FEF2575_pre'].std()
print(SD_G21_Boys_FEF2575)
Min_G21_Boys_FEF2575 = G21_Boys['FEF2575_pre'].min()
print(Min_G21_Boys_FEF2575)
Max_G21_Boys_FEF2575 = G21_Boys['FEF2575_pre'].max()
print(Max_G21_Boys_FEF2575)
# %%
Mean_G21_Girls_FEF2575 = G21_Girls['FEF2575_pre'].mean()
print(Mean_G21_Girls_FEF2575)
SD_G21_Girls_FEF2575 = G21_Girls['FEF2575_pre'].std()
print(SD_G21_Girls_FEF2575)
Min_G21_Girls_FEF2575 = G21_Girls['FEF2575_pre'].min()
print(Min_G21_Girls_FEF2575)
Max_G21_Girls_FEF2575 = G21_Girls['FEF2575_pre'].max()
print(Max_G21_Girls_FEF2575)
# %%
#FEV1FVC_pre:
stats.ttest_ind(G21_Boys["FEV1FVC_pre"],G21_Girls["FEV1FVC_pre"], equal_var = True)
#%%
sns.distplot( G21_Boys["FEV1FVC_pre"] , color="skyblue", label="Boys FVC")
sns.distplot( G21_Girls["FEV1FVC_pre"] , color="red", label="Girls FVC")
#%%
sns.boxplot( x=G21["Gender"], y=G21["FEV1FVC_pre"] )
# %%
Mean_G21_Boys_FEV1FVC = G21_Boys['FEV1FVC_pre'].mean()
print(Mean_G21_Boys_FEV1FVC)
SD_G21_Boys_FEV1FVC = G21_Boys['FEV1FVC_pre'].std()
print(SD_G21_Boys_FEV1FVC)
Min_G21_Boys_FEV1FVC = G21_Boys['FEV1FVC_pre'].min()
print(Min_G21_Boys_FEV1FVC)
Max_G21_Boys_FEV1FVC = G21_Boys['FEV1FVC_pre'].max()
print(Max_G21_Boys_FEV1FVC)
# %%
Mean_G21_Girls_FEV1FVC = G21_Girls['FEV1FVC_pre'].mean()
print(Mean_G21_Girls_FEV1FVC)
SD_G21_Girls_FEV1FVC = G21_Girls['FEV1FVC_pre'].std()
print(SD_G21_Girls_FEV1FVC)
Min_G21_Girls_FEV1FVC = G21_Girls['FEV1FVC_pre'].min()
print(Min_G21_Girls_FEV1FVC)
Max_G21_Girls_FEV1FVC = G21_Girls['FEV1FVC_pre'].max()
print(Max_G21_Girls_FEV1FVC)
# %%
#Comparissons between Derivation and Validation cohorts:
#Two-sample t-test comparissons between ARIA and G21 participants:
#Age:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["Idade"] ).set(title='Age distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='Age')
# In[]
sns.distplot( ARIA["Idade"] , color="skyblue")
sns.distplot( G21["Idade"] , color="red").set(xlabel="ARIA Age = Blue G21 Age = Red")
# In[]
stats.ttest_ind(ARIA["Idade"],G21["Idade"], equal_var = True)
#In[]
print(G21['Idade'].mean())
print(G21['Idade'].std())
print(G21['Idade'].min())
print(G21['Idade'].max())
# In[]
print(ARIA['Idade'].mean())
print(ARIA['Idade'].std())
print(ARIA['Idade'].min())
print(ARIA['Idade'].max())
# %%
#Height:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["Altura_cm"] ).set(title='Height distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='Height')
# In[]
sns.distplot( ARIA["Altura_cm"] , color="skyblue")
sns.distplot( G21["Altura_cm"] , color="red").set(xlabel="ARIA = Blue G21 = Red", title="Height distribution by Project")
# In[]
stats.ttest_ind(ARIA["Altura_cm"],G21["Altura_cm"], equal_var = True)
#In[]
print(G21['Altura_cm'].mean())
print(G21['Altura_cm'].std())
print(G21['Altura_cm'].min())
print(G21['Altura_cm'].max())
# In[]
print(ARIA['Altura_cm'].mean())
print(ARIA['Altura_cm'].std())
print(ARIA['Altura_cm'].min())
print(ARIA['Altura_cm'].max())
# %%
#Weight:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["Peso"] ).set(title='Weight distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='Weight')
# In[]
sns.distplot( ARIA["Peso"] , color="skyblue")
sns.distplot( G21["Peso"] , color="red").set(xlabel="ARIA = Blue G21 = Red", title="Weight distribution by Project")
# In[]
stats.ttest_ind(ARIA["Peso"],G21["Peso"], equal_var = True)
#In[]
print(G21['Peso'].mean())
print(G21['Peso'].std())
print(G21['Peso'].min())
print(G21['Peso'].max())
# In[]
print(ARIA['Peso'].mean())
print(ARIA['Peso'].std())
print(ARIA['Peso'].min())
print(ARIA['Peso'].max())
# %%
#BMI:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["BMI"] ).set(title='BMI distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='BMI')
# In[]
sns.distplot( ARIA["BMI"] , color="skyblue")
sns.distplot( G21["BMI"] , color="red").set(xlabel="Blue = ARIA Red = G21", title="BMI distribution by Project")
# In[]
stats.ttest_ind(ARIA["BMI"],G21["BMI"], equal_var = True)
#In[]
print(G21['BMI'].mean())
print(G21['BMI'].std())
print(G21['BMI'].min())
print(G21['BMI'].max())
# In[]
print(ARIA['BMI'].mean())
print(ARIA['BMI'].std())
print(ARIA['BMI'].min())
print(ARIA['BMI'].max())
# %%
#FVC:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["FVC_pre"] ).set(title='FVC distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='FVC (L)')
# In[]
sns.distplot( ARIA["FVC_pre"] , color="skyblue")
sns.distplot( G21["FVC_pre"] , color="red").set(xlabel="Blue = ARIA Red = G21", title="FVC distribution by Project")
# In[]
stats.ttest_ind(ARIA["FVC_pre"],G21["FVC_pre"], equal_var = True)
#In[]
print(G21['FVC_pre'].mean())
print(G21['FVC_pre'].std())
print(G21['FVC_pre'].min())
print(G21['FVC_pre'].max())
# In[]
print(ARIA['FVC_pre'].mean())
print(ARIA['FVC_pre'].std())
print(ARIA['FVC_pre'].min())
print(ARIA['FVC_pre'].max())
# %%
#FEV1:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["FEV1_pre"] ).set(title='FEV1 distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='FEV1 (L)')
# In[]
sns.distplot( ARIA["FEV1_pre"] , color="skyblue")
sns.distplot( G21["FEV1_pre"] , color="red").set(xlabel="Blue = ARIA Red = G21", title="FEV1 distribution by Project")
# In[]
stats.ttest_ind(ARIA["FEV1_pre"],G21["FEV1_pre"], equal_var = True)
#In[]
print(G21['FEV1_pre'].mean())
print(G21['FEV1_pre'].std())
print(G21['FEV1_pre'].min())
print(G21['FEV1_pre'].max())
# In[]
print(ARIA['FEV1_pre'].mean())
print(ARIA['FEV1_pre'].std())
print(ARIA['FEV1_pre'].min())
print(ARIA['FEV1_pre'].max())
# %%
#FEF2575:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["FEF2575_pre"] ).set(title='FEF25/75 distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='FEF25/75 (L/s)')
# In[]
sns.distplot( ARIA["FEF2575_pre"] , color="skyblue")
sns.distplot( G21["FEF2575_pre"] , color="red").set(xlabel="Blue = ARIA Red = G21", title="FEF25/75 distribution by Project")
# In[]
stats.ttest_ind(ARIA["FEF2575_pre"],G21["FEF2575_pre"], equal_var = True)
#In[]
print(G21['FEF2575_pre'].mean())
print(G21['FEF2575_pre'].std())
print(G21['FEF2575_pre'].min())
print(G21['FEF2575_pre'].max())
# In[]
print(ARIA['FEF2575_pre'].mean())
print(ARIA['FEF2575_pre'].std())
print(ARIA['FEF2575_pre'].min())
print(ARIA['FEF2575_pre'].max())
# %%
#FEV1/FVC:
sns.boxplot( x=SEL2020["Project"], y=SEL2020["FEV1FVC_pre"] ).set(title='FEV1/FVC distribution by Project', xlabel='Project: 0 = ARIA, 1 = G21', ylabel='FEV1/FVC')
# In[]
sns.distplot( ARIA["FEV1FVC_pre"] , color="skyblue")
sns.distplot( G21["FEV1FVC_pre"] , color="red").set(xlabel="Blue = ARIA Red = G21", title="FEV1/FVC distribution by Project")
# In[]
stats.ttest_ind(ARIA["FEV1FVC_pre"],G21["FEV1FVC_pre"], equal_var = True)
#In[]
print(G21['FEV1FVC_pre'].mean())
print(G21['FEV1FVC_pre'].std())
print(G21['FEV1FVC_pre'].min())
print(G21['FEV1FVC_pre'].max())
# In[]
print(ARIA['FEV1FVC_pre'].mean())
print(ARIA['FEV1FVC_pre'].std())
print(ARIA['FEV1FVC_pre'].min())
print(ARIA['FEV1FVC_pre'].max())
# %%
#Levene's test of variance between ARIA and G21:
#Age:
print(scipy.stats.levene(ARIA["Idade"], G21["Idade"]))
#Height:
print(scipy.stats.levene(ARIA["Altura_cm"], G21["Altura_cm"]))
#Weight:
print(scipy.stats.levene(ARIA["Peso"], G21["Peso"]))
#BMI:
print(scipy.stats.levene(ARIA["BMI"], G21["BMI"]))
#FVC:
print(scipy.stats.levene(ARIA["FVC_pre"], G21["FVC_pre"]))
#FEV1:
print(scipy.stats.levene(ARIA["FEV1_pre"], G21["FEV1_pre"]))
#FEF2575:
print(scipy.stats.levene(ARIA["FEF2575_pre"], G21["FEF2575_pre"]))
#FEV1FVC:
print(scipy.stats.levene(ARIA["FEV1FVC_pre"], G21["FEV1FVC_pre"]))
# %%
#Pearson's correlation coefficient ARIA cohort between antropometric and lung function parameters with graphs:
#FVC and Age ARIA:
print(scipy.stats.pearsonr(ARIA["FVC_pre"],ARIA["Idade"]))
sns.regplot(x=ARIA["Idade"], y=ARIA["FVC_pre"], fit_reg=True)
sns.lmplot( x="Idade", y="FVC_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Idade"], y=ARIA["FVC_pre"], kind='kde')
# %%
#FVC and Age ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FVC_pre"],ARIA_Boys["Idade"]))
sns.regplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FVC_pre"], kind='kde')
# %%
#FVC and Age ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FVC_pre"],ARIA_Girls["Idade"]))
sns.regplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FVC_pre"], kind='kde')
# %%
#FVC and Height ARIA:
print(scipy.stats.pearsonr(ARIA["FVC_pre"],ARIA["Altura_cm"]))
sns.regplot(x=ARIA["Altura_cm"], y=ARIA["FVC_pre"], fit_reg=True)
sns.lmplot( x="Altura_cm", y="FVC_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Altura_cm"], y=ARIA["FVC_pre"], kind='kde')
# %%
#FVC and Height ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FVC_pre"],ARIA_Boys["Altura_cm"]))
sns.regplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FVC_pre"], kind='kde')
#%%
#FVC and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FVC_pre"],ARIA_Girls["Altura_cm"]))
sns.regplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FVC_pre"], kind='kde')
# %%
#FVC and Weight ARIA:
print(scipy.stats.pearsonr(ARIA["FVC_pre"],ARIA["Peso"]))
sns.regplot(x=ARIA["Peso"], y=ARIA["FVC_pre"], fit_reg=True)
sns.lmplot( x="Peso", y="FVC_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Peso"], y=ARIA["FVC_pre"], kind='kde')
#%%
#FVC and Weight ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FVC_pre"],ARIA_Boys["Peso"]))
sns.regplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FVC_pre"], kind='kde')
#%%
#FVC and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FVC_pre"],ARIA_Girls["Peso"]))
sns.regplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FVC_pre"], kind='kde')
# %%
#FVC and BMI ARIA:
print(scipy.stats.pearsonr(ARIA["FVC_pre"],ARIA["BMI"]))
sns.regplot(x=ARIA["BMI"], y=ARIA["FVC_pre"], fit_reg=True)
sns.lmplot( x="BMI", y="FVC_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["BMI"], y=ARIA["FVC_pre"], kind='kde')
#%%
#FVC and BMI ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FVC_pre"],ARIA_Boys["BMI"]))
sns.regplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FVC_pre"], kind='kde')
#%%
#FVC and BMI ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FVC_pre"],ARIA_Girls["BMI"]))
sns.regplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FVC_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FVC_pre"], kind='kde')
# %%
#FEV1 and Age ARIA:
print(scipy.stats.pearsonr(ARIA["FEV1_pre"],ARIA["Idade"]))
sns.regplot(x=ARIA["Idade"], y=ARIA["FEV1_pre"], fit_reg=True)
sns.lmplot( x="Idade", y="FEV1_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Idade"], y=ARIA["FEV1_pre"], kind='kde')
# %%
#FEV1 and Age ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEV1_pre"],ARIA_Boys["Idade"]))
sns.regplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FEV1_pre"], kind='kde')
# %%
#FEV1 and Age ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEV1_pre"],ARIA_Girls["Idade"]))
sns.regplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FEV1_pre"], kind='kde')
# %%
#FEV1 and Height ARIA:
print(scipy.stats.pearsonr(ARIA["FEV1_pre"],ARIA["Altura_cm"]))
sns.regplot(x=ARIA["Altura_cm"], y=ARIA["FEV1_pre"], fit_reg=True)
sns.lmplot( x="Altura_cm", y="FEV1_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Altura_cm"], y=ARIA["FEV1_pre"], kind='kde')
# %%
#FEV1 and Height ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEV1_pre"],ARIA_Boys["Altura_cm"]))
sns.regplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FEV1_pre"], kind='kde')
#%%
#FEV1 and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEV1_pre"],ARIA_Girls["Altura_cm"]))
sns.regplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FEV1_pre"], kind='kde')
# %%
#FEV1 and Weight ARIA:
print(scipy.stats.pearsonr(ARIA["FEV1_pre"],ARIA["Peso"]))
sns.regplot(x=ARIA["Peso"], y=ARIA["FEV1_pre"], fit_reg=True)
sns.lmplot( x="Peso", y="FEV1_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Peso"], y=ARIA["FEV1_pre"], kind='kde')
#%%
#FEV1 and Weight ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEV1_pre"],ARIA_Boys["Peso"]))
sns.regplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FEV1_pre"], kind='kde')
#%%
#FEV1 and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEV1_pre"],ARIA_Girls["Peso"]))
sns.regplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FEV1_pre"], kind='kde')
# %%
#FEV1 and BMI ARIA:
print(scipy.stats.pearsonr(ARIA["FEV1_pre"],ARIA["BMI"]))
sns.regplot(x=ARIA["BMI"], y=ARIA["FEV1_pre"], fit_reg=True)
sns.lmplot( x="BMI", y="FEV1_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["BMI"], y=ARIA["FEV1_pre"], kind='kde')
#%%
#FEV1 and BMI ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEV1_pre"],ARIA_Boys["BMI"]))
sns.regplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FEV1_pre"], kind='kde')
#%%
#FEV1 and BMI ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEV1_pre"],ARIA_Girls["BMI"]))
sns.regplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FEV1_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FEV1_pre"], kind='kde')
# %%
#FEF25/75 and Age ARIA:
print(scipy.stats.pearsonr(ARIA["FEF2575_pre"],ARIA["Idade"]))
sns.regplot(x=ARIA["Idade"], y=ARIA["FEF2575_pre"], fit_reg=True)
sns.lmplot( x="Idade", y="FEF2575_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Idade"], y=ARIA["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and Age ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEF2575_pre"],ARIA_Boys["Idade"]))
sns.regplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Idade"], y=ARIA_Boys["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and Age ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEF2575_pre"],ARIA_Girls["Idade"]))
sns.regplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Idade"], y=ARIA_Girls["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and Height ARIA:
print(scipy.stats.pearsonr(ARIA["FEF2575_pre"],ARIA["Altura_cm"]))
sns.regplot(x=ARIA["Altura_cm"], y=ARIA["FEF2575_pre"], fit_reg=True)
sns.lmplot( x="Altura_cm", y="FEF2575_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Altura_cm"], y=ARIA["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and Height ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEF2575_pre"],ARIA_Boys["Altura_cm"]))
sns.regplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FEF2575_pre"], kind='kde')
#%%
#FEF25/75 and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEF2575_pre"],ARIA_Girls["Altura_cm"]))
sns.regplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and Weight ARIA:
print(scipy.stats.pearsonr(ARIA["FEF2575_pre"],ARIA["Peso"]))
sns.regplot(x=ARIA["Peso"], y=ARIA["FEF2575_pre"], fit_reg=True)
sns.lmplot( x="Peso", y="FEF2575_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["Peso"], y=ARIA["FEF2575_pre"], kind='kde')
#%%
#FEF25/75 and Weight ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEF2575_pre"],ARIA_Boys["Peso"]))
sns.regplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["Peso"], y=ARIA_Boys["FEF2575_pre"], kind='kde')
#%%
#FEF25/75 and Height ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEF2575_pre"],ARIA_Girls["Peso"]))
sns.regplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["Peso"], y=ARIA_Girls["FEF2575_pre"], kind='kde')
# %%
#FEF25/75 and BMI ARIA:
print(scipy.stats.pearsonr(ARIA["FEF2575_pre"],ARIA["BMI"]))
sns.regplot(x=ARIA["BMI"], y=ARIA["FEF2575_pre"], fit_reg=True)
sns.lmplot( x="BMI", y="FEF2575_pre", data=ARIA, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=ARIA["BMI"], y=ARIA["FEF2575_pre"], kind='kde')
#%%
#FEF25/75 and BMI ARIA_Boys:
print(scipy.stats.pearsonr(ARIA_Boys["FEF2575_pre"],ARIA_Boys["BMI"]))
sns.regplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Boys["BMI"], y=ARIA_Boys["FEF2575_pre"], kind='kde')
#%%
#FEF25/75 and BMI ARIA_Girls:
print(scipy.stats.pearsonr(ARIA_Girls["FEF2575_pre"],ARIA_Girls["BMI"]))
sns.regplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FEF2575_pre"], fit_reg=True)
sns.jointplot(x=ARIA_Girls["BMI"], y=ARIA_Girls["FEF2575_pre"], kind='kde')
# %%
#FEV1/FVC and Age ARIA:
def pearsoncc (x,y):
 print (scipy.stats.pearsonr(x,y))

def boxplotall (x,y):
    return (sns.regplot(x, y, fit_reg=True))

def boxplotgender (x,y,z):
    return (sns.lmplot( x, y, z, fit_reg=True, hue='Gender', legend=False))

def boxplotkde (x,y):
    return (sns.jointplot(x, y, kind='kde'))
#%%
pearsoncc (ARIA["FEV1FVC_pre"],ARIA["Idade"])
boxplotall(x=ARIA["Idade"], y=ARIA["FEV1FVC_pre"])
boxplotgender(x="Idade", y="FEV1FVC_pre", z=ARIA)
boxplotkde (ARIA["Idade"], ARIA["FEV1FVC_pre"])
# %%
#FEV1/FVC and Age ARIA_Boys:
pearsoncc (ARIA_Boys["FEV1FVC_pre"],ARIA_Boys["Idade"])
boxplotall(x=ARIA_Boys["Idade"], y=ARIA_Boys["FEV1FVC_pre"])
boxplotkde (ARIA_Boys["Idade"], ARIA_Boys["FEV1FVC_pre"])
# %%
#FEV1/FVC and Age ARIA_Girls:
pearsoncc (ARIA_Girls["FEV1FVC_pre"],ARIA_Girls["Idade"])
boxplotall(x=ARIA_Girls["Idade"], y=ARIA_Girls["FEV1FVC_pre"])
boxplotkde (ARIA_Girls["Idade"], ARIA_Girls["FEV1FVC_pre"])
# %%
#FEV1/FVC and Height ARIA:
pearsoncc (ARIA["FEV1FVC_pre"],ARIA["Altura_cm"])
boxplotall(x=ARIA["Altura_cm"], y=ARIA["FEV1FVC_pre"])
boxplotgender(x="Altura_cm", y="FEV1FVC_pre", z=ARIA)
boxplotkde (ARIA["Altura_cm"], ARIA["FEV1FVC_pre"])
# %%
#FEV1/FVC and Height ARIA_Boys:
pearsoncc (ARIA_Boys["FEV1FVC_pre"],ARIA_Boys["Altura_cm"])
boxplotall(x=ARIA_Boys["Altura_cm"], y=ARIA_Boys["FEV1FVC_pre"])
boxplotkde (ARIA_Boys["Altura_cm"], ARIA_Boys["FEV1FVC_pre"])
# %%
#FEV1/FVC and Height ARIA_Girls:
pearsoncc (ARIA_Girls["FEV1FVC_pre"],ARIA_Girls["Altura_cm"])
boxplotall(x=ARIA_Girls["Altura_cm"], y=ARIA_Girls["FEV1FVC_pre"])
boxplotkde (ARIA_Girls["Altura_cm"], ARIA_Girls["FEV1FVC_pre"])
# %%
#FEV1/FVC and Weight ARIA:
pearsoncc (ARIA["FEV1FVC_pre"],ARIA["Peso"])
boxplotall(x=ARIA["Peso"], y=ARIA["FEV1FVC_pre"])
boxplotgender(x="Peso", y="FEV1FVC_pre", z=ARIA)
boxplotkde (ARIA["Peso"], ARIA["FEV1FVC_pre"])
# %%
#FEV1/FVC and Weight ARIA_Boys:
pearsoncc (ARIA_Boys["FEV1FVC_pre"],ARIA_Boys["Peso"])
boxplotall(x=ARIA_Boys["Peso"], y=ARIA_Boys["FEV1FVC_pre"])
boxplotkde (ARIA_Boys["Peso"], ARIA_Boys["FEV1FVC_pre"])
# %%
#FEV1/FVC and Weight ARIA_Girls:
pearsoncc (ARIA_Girls["FEV1FVC_pre"],ARIA_Girls["Peso"])
boxplotall(x=ARIA_Girls["Peso"], y=ARIA_Girls["FEV1FVC_pre"])
boxplotkde (ARIA_Girls["Peso"], ARIA_Girls["FEV1FVC_pre"])
# %%
#FEV1/FVC and BMI ARIA:
pearsoncc (ARIA["FEV1FVC_pre"],ARIA["BMI"])
boxplotall(x=ARIA["BMI"], y=ARIA["FEV1FVC_pre"])
boxplotgender(x="BMI", y="FEV1FVC_pre", z=ARIA)
boxplotkde (ARIA["BMI"], ARIA["FEV1FVC_pre"])
# %%
#FEV1/FVC and BMI ARIA_Boys:
pearsoncc (ARIA_Boys["FEV1FVC_pre"],ARIA_Boys["BMI"])
boxplotall(x=ARIA_Boys["BMI"], y=ARIA_Boys["FEV1FVC_pre"])
boxplotkde (ARIA_Boys["BMI"], ARIA_Boys["FEV1FVC_pre"])
# %%
#FEV1/FVC and BMI ARIA_Girls:
pearsoncc (ARIA_Girls["FEV1FVC_pre"],ARIA_Girls["BMI"])
boxplotall(x=ARIA_Girls["BMI"], y=ARIA_Girls["FEV1FVC_pre"])
boxplotkde (ARIA_Girls["BMI"], ARIA_Girls["FEV1FVC_pre"])
# %%
#Spirometry predictive models:
#Multiple Linear Regression:
#FVC Boys all variables:
X = ARIA_Boys[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Boys['FVC_pre']
# %%
model = ols("y ~ X", ARIA_Boys).fit()
print(model.summary())
# %%
#FEV1 Boys all variables:
X = ARIA_Boys[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Boys['FEV1_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Boys).fit()
print(model.summary())
# %%
#FEF2575 Boys all variables:
X = ARIA_Boys[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Boys['FEF2575_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Boys).fit()
print(model.summary())
# %%
#FEV1FVC Boys all variables:
X = ARIA_Boys[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Boys['FEV1FVC_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Boys).fit()
print(model.summary())
# %%
#FVC Girls all variables:
X = ARIA_Girls[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Girls['FVC_pre']
# %%
model = ols("y ~ X", ARIA_Girls).fit()
print(model.summary())
# %%
#FEV1 Girls all variables:
X = ARIA_Girls[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Girls['FEV1_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Girls).fit()
print(model.summary())
# %%
#FEF2575 Girls all variables:
X = ARIA_Girls[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Girls['FEF2575_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Girls).fit()
print(model.summary())
# %%
#FEV1FVC Girls all variables:
X = ARIA_Girls[['Idade','Altura_cm', 'Peso', 'BMI']]
y = ARIA_Girls['FEV1FVC_pre']
# %%np.sum(result.resid)
model = ols("y ~ X", ARIA_Girls).fit()
print(model.summary())
# %%
#Bland-Altman plot to access degree of agreement between measured and predicted values:
#FVC:
G21.info()
sm.graphics.mean_diff_plot(G21["FVC_pre"], G21["FVC_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21["FVC_GLI"] = pd.to_numeric(G21["FVC_GLI"])
sm.graphics.mean_diff_plot(G21["FVC_pre"], G21["FVC_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FVC Boys:
G21_Boys.info()
sm.graphics.mean_diff_plot(G21_Boys["FVC_pre"], G21_Boys["FVC_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Boys["FVC_GLI"] = pd.to_numeric(G21_Boys["FVC_GLI"])
sm.graphics.mean_diff_plot(G21_Boys["FVC_pre"], G21_Boys["FVC_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FVC Girls:
G21_Girls.info()
sm.graphics.mean_diff_plot(G21_Girls["FVC_pre"], G21_Girls["FVC_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Girls["FVC_GLI"] = pd.to_numeric(G21_Girls["FVC_GLI"])
sm.graphics.mean_diff_plot(G21_Girls["FVC_pre"], G21_Girls["FVC_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FEV1:
G21.info()
sm.graphics.mean_diff_plot(G21["FEV1_pre"], G21["FEV_1_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21["FEV1_GLI"] = pd.to_numeric(G21["FEV1_GLI"])
sm.graphics.mean_diff_plot(G21["FEV1_pre"], G21["FEV1_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)

# %%
#FEV1 Boys:
G21_Boys.info()
sm.graphics.mean_diff_plot(G21_Boys["FEV1_pre"], G21_Boys["FEV_1_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Boys["FEV1_GLI"] = pd.to_numeric(G21_Boys["FEV1_GLI"])
sm.graphics.mean_diff_plot(G21_Boys["FEV1_pre"], G21_Boys["FEV1_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FEV1 Girls:
G21_Girls.info()
sm.graphics.mean_diff_plot(G21_Girls["FEV1_pre"], G21_Girls["FEV_1_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Girls["FEV1_GLI"] = pd.to_numeric(G21_Girls["FEV1_GLI"])
sm.graphics.mean_diff_plot(G21_Girls["FEV1_pre"], G21_Girls["FEV1_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FEF25/75:
G21.info()
sm.graphics.mean_diff_plot(G21["FEF2575_pre"], G21["FEF2575_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21["FEF2575_GLI"] = pd.to_numeric(G21["FEF2575_GLI"])
sm.graphics.mean_diff_plot(G21["FEF2575_pre"], G21["FEF2575_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FEF2575 Boys:
G21_Boys.info()
sm.graphics.mean_diff_plot(G21_Boys["FEF2575_pre"], G21_Boys["FEF2575_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Boys["FEF2575_GLI"] = pd.to_numeric(G21_Boys["FEF2575_GLI"])
sm.graphics.mean_diff_plot(G21_Boys["FEF2575_pre"], G21_Boys["FEF2575_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#FEF2575 Girls:
G21_Girls.info()
sm.graphics.mean_diff_plot(G21_Girls["FEF2575_pre"], G21_Girls["FEF2575_ARIA"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
#%%
G21_Girls["FEF2575_GLI"] = pd.to_numeric(G21_Girls["FEF2575_GLI"])
sm.graphics.mean_diff_plot(G21_Girls["FEF2575_pre"], G21_Girls["FEF2575_GLI"], sd_limit=1.96, ax=None, scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None)
# %%
#Pearson's correlation Coefficient between measured and predicted values:
#FVC:
print(scipy.stats.pearsonr(G21["FVC_pre"],G21["FVC_ARIA"]))
print(scipy.stats.pearsonr(G21["FVC_pre"],G21["FVC_GLI"]))
print(scipy.stats.pearsonr(G21_Boys["FVC_pre"],G21_Boys["FVC_ARIA"]))
print(scipy.stats.pearsonr(G21_Boys["FVC_pre"],G21_Boys["FVC_GLI"]))
print(scipy.stats.pearsonr(G21_Girls["FVC_pre"],G21_Girls["FVC_ARIA"]))
print(scipy.stats.pearsonr(G21_Girls["FVC_pre"],G21_Girls["FVC_GLI"]))
#%%
#FEV1:
print(scipy.stats.pearsonr(G21["FEV1_pre"],G21["FEV_1_ARIA"]))
print(scipy.stats.pearsonr(G21["FEV1_pre"],G21["FEV1_GLI"]))
print(scipy.stats.pearsonr(G21_Boys["FEV1_pre"],G21_Boys["FEV_1_ARIA"]))
print(scipy.stats.pearsonr(G21_Boys["FEV1_pre"],G21_Boys["FEV1_GLI"]))
print(scipy.stats.pearsonr(G21_Girls["FEV1_pre"],G21_Girls["FEV_1_ARIA"]))
print(scipy.stats.pearsonr(G21_Girls["FEV1_pre"],G21_Girls["FEV1_GLI"]))
#%%
#FEF25/75:
print(scipy.stats.pearsonr(G21["FEF2575_pre"],G21["FEF2575_ARIA"]))
print(scipy.stats.pearsonr(G21["FEF2575_pre"],G21["FEF2575_GLI"]))
print(scipy.stats.pearsonr(G21_Boys["FEF2575_pre"],G21_Boys["FEF2575_ARIA"]))
print(scipy.stats.pearsonr(G21_Boys["FEF2575_pre"],G21_Boys["FEF2575_GLI"]))
print(scipy.stats.pearsonr(G21_Girls["FEF2575_pre"],G21_Girls["FEF2575_ARIA"]))
print(scipy.stats.pearsonr(G21_Girls["FEF2575_pre"],G21_Girls["FEF2575_GLI"]))
#%% 
#Sample graphs (to adapt as needed):
sns.regplot(x=G21["FVC_pre"], y=G21["FVC_ARIA"], fit_reg=True)
sns.lmplot( x="FVC_pre", y="FVC_ARIA", data=G21, fit_reg=True, hue='Gender', legend=False)
sns.jointplot(x=G21["FVC_pre"], y=G21["FVC_ARIA"], kind='kde')
# %%
