import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel('data/train-test.xlsx')
p = ['Function', 'Corner', 'Entrance', 'Contrast', 'FaceArea', 'Elevator','Familarity']
p_drop = ['Salience', 'Color', 'Name', 'Intersection']
x = df.drop(p_drop, axis=1)
x_train_column_name = list(x.columns)
s = StandardScaler()
x = s.fit_transform(np.array(x))
y = df['Salience']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

# The importance of RF model features under the optimal parameter combination is obtained.
RF = RandomForestRegressor(n_estimators=6,max_features=4,max_depth=14,min_samples_split=6,random_state=75)
RF_m = RF.fit(x_train, y_train)
# The importance of calculating independent variables
ALL_Index_Importances = list(RF_m.feature_importances_)
all_index_name = []
for i, name in enumerate(x_train_column_name):
    all_index_name.append((name,ALL_Index_Importances[i]))
sorted_all_index_name = sorted(all_index_name, key=lambda x: x[1], reverse=False)
index_importance = [item[1] for item in sorted_all_index_name]
index_name = [item[0] for item in sorted_all_index_name]
print(sorted_all_index_name)
# print(index_importance)
plt.barh(range(len(index_importance)),index_importance)
plt.yticks(range(len(index_importance)),index_name)
plt.ylabel('POI significant evaluation index')
plt.xlabel('weight value')
plt.show()