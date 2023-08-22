#Question 3a

import numpy as np


A = np.array([[2,1,2],
     [1,-2,1],
     [1,2,3],
     [1,1,1]])

b = np.array([6,
     1,
     5,
     2]).reshape(4,1)

x = np.linalg.lstsq(A,b,rcond=None)

aproximated_x = np.array(x[0]).reshape(3,1)


print("This is the best approximation in the least square sense for the system Ax = b : ")
print(aproximated_x)



#Question 3b

minimal_objective = x[1][0]

print("This is the minimal objective (loss) value : ")
print(minimal_objective)

#Question 3c
r = np.dot(A, aproximated_x) - b

Atr = np.dot(A.transpose(), r)

print("This is the residual of the least square system :")
print(r)
print('----------------------------------')
print("Here we can see that for the multiplication if A.transpose with the residual computes a 'numerical zero' :")
print(Atr)

#Question 3d
W = np.array([[1,0,0,0],
              [0,1000,0,0],
              [0,0,1,0],
              [0,0,0,1]])

Atw = np.array(np.dot(A.transpose(), W))
AtwA = np.array(np.dot(Atw, A))
Atwb = np.array(np.dot(Atw, b))
in_AtwA = np.linalg.inv(AtwA)

weighted_appr_x = np.dot(in_AtwA, Atwb)
weighted_r = r = np.dot(A, weighted_appr_x) - b

print("This is the best approximation in the least square sense with weights for the system Ax = b : ")
print(weighted_appr_x)
print('----------------------------------')
print("This is the residual for this approximation: ")
print(weighted_r)
print("We can see thar |r2| < 1/1000. i.e the second equatiotion of the solution is almost exactly satisfied.")


#Question 3e
T = np.array([[0.5,0,0],
              [0,0.5,0],
              [0,0,0.5]])

AtA = np.array(np.dot(A.transpose(), A))
AtwAT = AtA + T
in_AtwAT = np.linalg.inv(AtwAT)
Atb =  np.array(np.dot(A.transpose(), b))
Tik_x = np.array(np.dot(in_AtwAT, Atb))

print("This is the best approximation in the least square sense with the Tikhonov regularization for the system Ax = b : ")
print(Tik_x)


#Question 4b
from sympy import symbols, solve
from pprint import pprint
M = np.array([[5,6,7,8],
              [1,3,5,4],
              [1,0.5,4,2],
              [3,4,3,1]])

B = np.array([[0.57,0.56,0.8,1],
              [1.5,4,6.7,4.9],
              [0.2,0.1,1,0.6],
              [11,30,26,10]])

d = symbols('d')
sym_D = [[0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]]

D = [[0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]]
    
for i in range(4):
    fi = 0
    for j in range(4):
        sym_D[i][j] = M[i][j]*d - B[i][j]
        fi = fi + (sym_D[i][j])**2
    d_fi = fi.diff(d)
    di = solve(d_fi)[0]
    D[i][i] = di   
print(D)


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


file_path = "/Users/lizgokhvat/Desktop/SPL Assignments/Intro-To-Data-Science/Assignment-1/insurData.csv"

df = pd.read_csv(file_path)

print('\nNumber of rows and columns in data set: (rows, columns) :', df.shape)
print('')

#Adding a column of 1's
df.insert(loc=0, column="s", value=1)
#Defining the charges in thousands( dividing by 1000)
df['charges'] = df['charges'].apply(lambda x: x / 1000)

#One-hot encoding
df = pd.get_dummies(df)

X = df[['s', 'age', 'sex_male', 'sex_female', 'bmi', 'children', 'smoker_yes', 'smoker_no',  'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]
y = df[['charges']]


for i in range(1,6):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True)

    print(f"experiments no. {i}")

    LS_x = np.linalg.lstsq(X_train,y_train)
    print("Least Squares Solution for train:")
    print(LS_x[0])

    M_train = np.dot(X_train, LS_x[0]) - y_train
    MSE_Train = (1/1070)*(np.linalg.norm(M_train))**2
    print("Train MSE:")
    print(MSE_Train)

    M_test = np.dot(X_test, LS_x[0]) - y_test
    MSE_Test = (1/268)*(np.linalg.norm(M_test))**2
    print("Test MSE:")
    print(MSE_Test)

    diff_MSE = np.abs(MSE_Test/MSE_Train)
    print("compare:")
    print(diff_MSE)

    print("-----------------------------------------------------------------------")

plt.hist(M_train.to_numpy(), bins=200)
plt.title('Error Distribution')
plt.xlabel('error')
plt.ylabel('error frequency')
plt.legend()
plt.show()



