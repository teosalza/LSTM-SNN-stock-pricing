import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import numpy as np
from sklearn.linear_model import LinearRegression
import time

dset = make_regression(n_samples=50,n_features=1,noise=8,random_state=3)
# dset = dset[0].reshape(100)
plt.scatter(dset[0],dset[1])
plt.show()

x = dset[0]
y = dset[1]

c = 0
m = 0
iteration = 100
learning_rate = 0.05

error_list = []
c_list = []
m_list = []

plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, c + m*x,color="red")
plt.scatter(x,y)

'''total_error = 1/n * sum(y_i - y_i^)
total_error = 1/n * sum(y_i - (c + m*x_i))^2 '''
for i in range(iteration):
    plt.title("Updating plot... iteration: {}".format(i))
    total_error = np.sum((y - (c + m*x).T)**2)/y.shape[0]
    grad_m = -2*np.sum(x.T*(y - (c + m*x).T))/y.shape[0]
    grad_c = -2*np.sum((y - (c + m*x).T))/y.shape[0]
    m = m - learning_rate*grad_m
    c = c - learning_rate*grad_c
    error_list.append(total_error)
    c_list.append(c)
    m_list.append(m)

    line1.set_xdata(x)
    line1.set_ydata(c + m*x)
    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(0.01)
    
plt.plot(np.arange(iteration),error_list)

print("Intercept (B0) value: {}".format(c))
print("B1 value: {}".format(m))
print("Last error: {}".format(error_list[-1]))