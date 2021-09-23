# Easy 2-D perceptron with iris data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = pd.DataFrame(iris["target"], columns =["target_names"])
df = pd.concat([x,y], axis = 1)

#drop some extreme value
df_t = df.drop(index = [41,106])

# make a training data frame
data_train = df_t[df_t["target_names"]!=1][['sepal length (cm)', 'sepal width (cm)','target_names']]
data_train['target_names'] = data_train['target_names'] -1
data_train.reset_index(drop=True, inplace =True)

sns.set_style('darkgrid')

# initiate c(w0, w1, w2)
w = np.array([0.1,0.1,0.1], dtype = 'float64')
x= data_train[['sepal length (cm)', 'sepal width (cm)']]
y = data_train["target_names"]

# my activation function
def sign(w, x):
    if np.dot(w[1:], x) > 0:
        return 1
    else:
        return -1

sign(w, x.loc[0]) == y[0]
error = 1
iterate = 0

while error != 0:
    error = 0
    # plot the original line with dash
    for i in range(len(x)):

        if sign(w, x.loc[i]) != y[i]:
            print(sign(w, x.loc[i]), y[i])
            sns.lmplot(data=data_train, x = 'sepal length (cm)' , y ='sepal width (cm)', hue ='target_names' , fit_reg = False )
            iterate += 1
            error += 1
            print("iterate: " + str(iterate))
            x_origin_vector = np.linspace(0, 10)
            y_origin_vector = -(w[1]/w[2]) * x_origin_vector - w[0]/w[2]
            plt.plot(x_origin_vector,y_origin_vector, 'c--', color = 'blue')

            # if the sign is not equal traget_names
            # set w(new) = w + (y-d)x

            w[1:] += (y[i]) * x.loc[i]
            w[0] +=  (y[i]) * w[0]

            # plot the update line 
            x_update_vector = np.linspace(0, 10)
            y_update_vector = -(w[1]/w[2]) * x_update_vector - w[0]/w[2]
            print(w)
            plt.plot(x_update_vector,y_update_vector, color = 'green')
  
            plt.xlim(0,10)
            plt.ylim(0,10)
            plt.show()