import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Draw the scatter plot between the real value and the predicted value.
def scatter_plot(TureValues,PredictValues,R2, train_mae, train_mse):
    plt.figure(figsize=(4, 3))
    plt.plot([0, 1], [0, 1], c='0', linewidth=1, linestyle=':', marker='', alpha=0.3)
    plt.scatter(TureValues, PredictValues, s=20, c='r', edgecolors='k', marker='o', alpha=0.8,
                label='R^2 = %.2f' % R2)
    plt.xlabel('The real value of POI significance')
    plt.ylabel('POI significance prediction value')
    # plt.legend()
    plt.text(0.33, 0.95, 'R^2 = {}\nMAE = {}\nMSE = {}'.format(train_R2, train_mae, train_mse), ha='right', va='top',
             transform=plt.gca().transAxes)
    plt.show()
# Draw the error histogram
def erro_plot(real,pred,bins):
    plt.figure(figsize=(4, 3))
    plt.hist(real-pred, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black', label='training sets')
    plt.axvline(x=0, color='red', linestyle='--', label='Error(0)')
    plt.legend()
#     plt.title('Error Histogram')
    plt.xlabel('Error = Real value - Predicted value')
    plt.ylabel('Frequency')
    plt.show()
    # plt.grid(True)

# The feature value training model is gradually reduced according to the feature importance ranking.
df = pd.read_excel('data/train-test.xlsx')
p = ['Function', 'Corner', 'Entrance', 'Contrast', 'FaceArea', 'Elevator','Familarity']
p_drop = ['Salience', 'Color', 'Name', 'Intersection']
cn = 0  # When cn is 0, all features ( 7 selected by correlation analysis ) are involved in training.
for i in range(0):
    p_drop.append(p[i])
x = df.drop(p_drop, axis=1)
max_features_n = len(list(x.columns))+1
print('training characteristics：', list(x.columns))
s = StandardScaler()
x = s.fit_transform(np.array(x))
y = df['Salience']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

# The model is constructed by the optimal parameter combination obtained by parameter optimization.
MLP = MLPRegressor(hidden_layer_sizes=(10, 9), learning_rate_init=0.1, max_iter=400, random_state=5, solver='sgd')
MLP_m = MLP.fit(x_train,y_train)
# Accuracy evaluation of training set
train_ypred = MLP_m.predict(x_train)
train_R2 = round(r2_score(y_train,train_ypred),3)
train_mae = round(mean_absolute_error(y_train,train_ypred),3)
train_mse = round(mean_squared_error(y_train,train_ypred),3)
print('training set(R2/MAE/MSE)：',train_R2,'/',train_mae,'/',train_mse)
scatter_plot(y_train,train_ypred,train_R2, train_mae, train_mse)
erro_plot(y_train,train_ypred,9)

# Accuracy evaluation of test set
test_ypred = MLP_m.predict(x_test)
test_R2 = round(r2_score(y_test,test_ypred),3)
test_mae = round(mean_absolute_error(y_test, test_ypred),3)
test_mse = round(mean_squared_error(y_test, test_ypred),3)
print('test set(R2/MAE/MSE)：',test_R2,'/',test_mae,'/',test_mse)
# # scatter_plot(y_test,y_pred,r2_score(y_test,y_pred))
print('The true value of the test set：', list(y_test))
print('Predicted value of test set：', test_ypred)