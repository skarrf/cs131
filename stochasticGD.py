from random import randint
y = [25, 37.5]
m = randint(1, 10) # w_old
x = [2, 3]
b = 0
rate = 0.1
n = 5 # number of for loops

print('Initial guess: ' + str(m))

for i in range(n):
    if (n % 2 == 0):
        prediction = m*x[0]
        loss = (y[0] - m*x[0]) ** 2
        derivative = 2*(y[0] - m*x[0]) * (0 - 1*x[0])
    if (n % 2 == 1):
        loss = (y[1] - m*x[1]) ** 2
        derivative = 2*(y[1] - m*x[1]) * (0 - 1*x[1])
    
    m = m - rate*derivative
    print('New m: ' + str(m))

# try m=8 and n=5



"""
from random import randint
y = 25
m = 12 # randint(1, 10) # w_old
x = 2
b = 0
rate = 0.1

print('Guess: ' + str(m))

prediction = m*x
print('Prediction: ' + str(prediction))

loss = (y - m*x) ** 2
print('Loss: ' + str(loss))

derivative = 2*(y - m*x) * (0 - 1*x)
print('Derivative: ' + str(derivative))

m = m - rate*derivative

print('New m: ' + str(m))
print('What we want: ' + str(y))
"""



"""
Ex. Y = mx, w2 = 25 (wanted output)
Guess or randomly initialize w_old
Compute prediction
Compute loss (y - mx)^2
Compute derivative of loss function (dloss/dw = 2(y - mx) * (0 - (1)(x)))
W_new = w_old - (learning rate) * (derivative of loss)
End: Stop if derivative = 0; Stop if w_new - w_old = 0.001
"""



"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import pandas
from random import randint
from sklearn.model_selection import train_test_split


train = pandas.read_csv('/Users/patrick/Desktop/cs131/train-1.csv')
test = pandas.read_csv('/Users/patrick/Desktop/cs131/test.csv')


# print(train.iloc[499, 68]) # iloc = integer location (indexes)
# y = 0 column; x = 1 -> 68 columns; y = 0 -> 499 rows
# y has 0-499 rows, and x has 1-68 columns
# train(usecols=range(1, 68))
x_train = train.drop('arousal', axis=1)
y_train = train['arousal']
x_test = test.drop('arousal', axis=1)
y_test = test['arousal']
# https://stackoverflow.com/questions/7588934/how-to-delete-columns-in-a-csv-file
# print(x_train)
# print(y_train)


# convert to arrays for matrix multiplication
# x_train = x_train.to_numpy()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
# google says: np.array(data_list)

# print(x_train)
# print(y_train)


### IDK WHAT THIS IS
x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)
### IDK WHAT THIS IS


# y = wx
# y = 500 by 1 matrix, x = 500 by 68 matrix, w should be 68 by 1 0-vector matrix
w = np.zeros((68)) # this is w_old
rate = 0.001
epochs = 100

# print(x_train.dot(w))
# google: vector = np.zeros((68))

# print(x_train.shape) # checks matrix dimension
# print(w.shape)
# print(y_train.shape)


training_loss_graph = []
validation_loss_graph = []

# google: indices = np.random.choice(len(data), size=num_rows, replace=False)
# print(x_train.sample(frac=450))
# print(np.random.choice(len(x_train), size=450, replace=False))

for i in range(0, epochs):
    # choose random rows for testing first
    random_row = np.random.choice(len(x_train), size=450, replace=False)
    random_x_train = x_train[random_row]
    random_y_train = y_train[random_row]
    
    # random_x_train = x_train.sample(frac=1)
    # random_y_train = y_train.sample(frac=1)

    for j in range(0, len(random_x_train)):
        prediction = random_x_train[j].dot(w) # xw
        # loss = (random_y_train[j] - prediction) ** 2 # (y - mx)^2
        derivative = 2 * (random_y_train[j] - prediction) * (0 - 1*random_x_train[j]) # 2(y - mx) * (0 - 1*x)

        # update w after epoch?
        w = w - rate*derivative # w_new
    
    # add the loss to the graph lists
    training_loss = (y_train - x_train.dot(w)) ** 2 # (y - mx)^2
    validation_loss = (y_test - x_test.dot(w)) ** 2 # (y - mx)^2
    
    training_loss_graph.append(training_loss)
    validation_loss_graph.append(validation_loss)

    print(f"Epoch {i}: Training Loss: {training_loss[i]:.4f}, Validation Loss: {validation_loss[i]:.4f}")



# Plot the training and validation loss curves

plt.plot(range(epochs), training_loss_graph, label='Train Loss')
plt.plot(range(epochs), validation_loss_graph, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves: Training vs Validation')
plt.legend()
plt.show()
"""