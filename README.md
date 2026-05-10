# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function compute Cost to generate the cost function.

3.Perform iterations of gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```

/*
Program to implement the linear regression using gradient descent.
Developed by: RITHIK RAM S 
RegisterNumber:  212224230229
*/
```

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("profit.csv")

print(data.head())

X = data['Population'].values
y = data['Profit'].values

m = len(y)

X = (X - X.mean()) / X.std()

theta0 = 0
theta1 = 0

learning_rate = 0.01
iterations = 1000

for i in range(iterations):

    y_pred = theta0 + theta1 * X

    d_theta0 = (1/m) * np.sum(y_pred - y)
    d_theta1 = (1/m) * np.sum((y_pred - y) * X)

    theta0 = theta0 - learning_rate * d_theta0
    theta1 = theta1 - learning_rate * d_theta1


print("Intercept (theta0):", theta0)
print("Slope (theta1):", theta1)


predictions = theta0 + theta1 * X


plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red')

plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Simple Linear Regression using Gradient Descent")

plt.show()
```

## Output:
<img width="896" height="147" alt="image" src="https://github.com/user-attachments/assets/82e1899a-a4d6-4de1-86d4-76e91ccfdd19" />





<img width="633" height="56" alt="image" src="https://github.com/user-attachments/assets/8a9d422e-fadd-4107-90b7-03f50f5ab760" />






<img width="763" height="575" alt="image" src="https://github.com/user-attachments/assets/17ec7ce6-30a8-4454-863c-3d11663947cb" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
