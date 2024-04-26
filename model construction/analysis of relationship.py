import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('data/train-test.xlsx')
x = df.drop(['Salience'], axis=1)

corr_matrix = x.corr(method='spearman')#Spearman's correlation coefficient
# corr_matrix = x.corr()#Pearson correlation coefficient

# Draw the correlation coefficient matrix diagram ( only show the lower triangular part )
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, vmin=-1, vmax=1, xticklabels=True)
plt.xticks(rotation=45)  # Rotate the x-axis label by 45 degrees
plt.title('')
plt.show()