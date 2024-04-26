import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# cross validation
def cv(xy_cv, model, num_folds):
    # Initialize the score list
    scores = []
    for fold in range(num_folds):
        val_x = []
        val_y = []
        train_x = []
        train_y = []
        for value in xy_cv:
            random_index = random.randint(0, len(value) - 1)
            val_x.append(value[random_index][1])
            val_y.append(value[random_index][0])
            for j in range(len(value)):
                if j != random_index:
                    train_x.append(value[j][1])
                    train_y.append(value[j][0])
        # training model
        model.fit(train_x, train_y)
        # Prediction validation set
        pred_y = model.predict(val_x)
        # Calculate the R2 score
        r2 = r2_score(val_y, pred_y)
        scores.append(round(r2, 3))

    return scores


# The feature value training model is gradually reduced according to the feature importance ranking.
df = pd.read_excel('data/train-test.xlsx')
p = ['Function', 'Corner', 'Entrance', 'Contrast', 'FaceArea', 'Elevator','Familarity']
p_drop = ['Salience', 'Color', 'Name', 'Intersection']
cn = 0 # When cn is 0, all features ( According to the correlation analysis, 7 items were selected ) are involved in training.
for i in range(0):
    p_drop.append(p[i])
x = df.drop(p_drop, axis=1)
max_features_n = len(list(x.columns))+1
print('training characteristics：', list(x.columns))
s = StandardScaler()
x = s.fit_transform(np.array(x))
y = df['Salience']
# Divide the test set and training set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
#
xy1 = []
xy2 = []
xy3 = []
xy4 = []
xy5 = []
for i, salience in enumerate(y):
    if 0.04 < salience < 0.11:
        xy1.append([salience, x[i]])
    elif 0.11 < salience < 0.21:
        xy2.append([salience, x[i]])
    elif 0.21 < salience < 0.40:
        xy3.append([salience, x[i]])
    elif 0.40 < salience < 0.60:
        xy4.append([salience, x[i]])
    elif 0.60 < salience < 0.90:
        xy5.append([salience, x[i]])
xy_cv = [xy1, xy2, xy3, xy4, xy5] #Data for cross validation

# The optimal parameter combination is obtained by cv training model.
m = [2, 3, 4, 5, 6, 7, 8, 9, 10]
n = [2, 3, 4, 5, 6, 7, 8, 9, 10]
random_state = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
learning_rate_init = [0.1, 0.01, 0.01]
max_iter = [100, 200, 300, 400]
activation = ['logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']

for i in m:
    for j in n:
        for rs in random_state:
            for mi in max_iter:
                for lri in learning_rate_init:
                    for act in activation:
                            for sol in solver:
                                mlp = MLPRegressor(activation=act, hidden_layer_sizes=(i, j), max_iter=mi,
                                                   learning_rate_init=lri,
                                                   random_state=rs, solver=sol)
                                try:
                                    scores = cv(xy_cv, mlp, 10)
                                    r2 = round(np.mean(scores), 3)
                                    print('\rThe average coefficient of determination of 10 training test results by cv：', r2, end='')
                                    if r2 >= 0.7:
                                        print('\n')
                                        print('All Coefficients of Determination：', scores)
                                        print('model structure：', mlp)
                                        print('Model training test results ( using training set and test set )：')
                                        best_model = mlp.fit(x_train, y_train)
                                        # training sets
                                        y_pred = best_model.predict(x_train)
                                        mae = round(mean_absolute_error(y_pred, y_train), 2)
                                        mse = round(mean_squared_error(y_pred, y_train), 2)
                                        print('training sets(R2/MAE/MSE)：', round(r2_score(y_train, y_pred), 2), '/', mae, '/', mse)
                                        # testing set
                                        y_pred = best_model.predict(x_test)
                                        mae = round(mean_absolute_error(y_pred, y_test), 2)
                                        mse = round(mean_squared_error(y_pred, y_test), 2)
                                        print('testing set(R2/MAE/MSE)：', round(r2_score(y_test, y_pred), 2), '/', mae, '/', mse)
                                        print('>—>—>—>—>—>—>—>—>—>—>—>—>—>>—>—>—>—>—>—>—>—>—>—>—>—>—>')
                                except:
                                    print('\nAn error has occurred. The error parameter is set to：', {'activation': act, 'hidden_layer_sizes': (i, j),
                                                              'max_iter': mi, 'learning_rate_init': lri,
                                                              'random_state': rs,
                                                              'solver': sol})



