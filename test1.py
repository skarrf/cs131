import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import pandas
from random import randint
from sklearn.model_selection import train_test_split
# said we could use sklearn to split

train = pandas.read_csv('/Users/patrick/Desktop/cs131/train-1.csv')
test = pandas.read_csv('/Users/patrick/Desktop/cs131/test.csv')

# y = 0 column; x = 1 -> 68 columns; y = 0 -> 499 rows
x_train = train.drop(['arousal'], axis=1)
y_train = train['arousal']

# x_validate = 
# y_validate = 

x_test = test.drop(['arousal'], axis=1)
y_test = test['arousal']
# https://stackoverflow.com/questions/7588934/how-to-delete-columns-in-a-csv-file

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
# google says: np.array(data_list) for matrix multiplication

w = 0 # w_old
rate = 0.000000001
epochs = 100

training_loss_graph = []
validation_loss_graph = []

for i in range(0, epochs):
    # split the data between train and validation first
    split_x_train, split_x_validate = train_test_split(x_train, test_size=0.1, random_state=42)
    split_y_train, split_y_validate = train_test_split(y_train, test_size=0.1, random_state=42)

    for j in range(0, len(split_x_train)):
        prediction = split_x_train[j].dot(w) # xw
        # loss = # (y - mx)^2
        derivative = 2 * (split_y_train[j] - prediction) * (0 - 1*split_x_train[j]) # 2(y - mx) * (0 - 1*x)

        w = w - rate*derivative # w_new
    
    # after epoch, calculate the loss then add to graph list
    training_loss = np.mean((split_y_train - split_x_train.dot(w)) ** 2) # (y - mx)^2
    validation_loss = np.mean((split_y_validate - split_x_validate.dot(w)) ** 2) # (y - mx)^2
    test_loss = np.mean((y_test - x_test.dot(w)) ** 2) # (y - mx)^2
    
    training_loss_graph.append(training_loss)
    validation_loss_graph.append(validation_loss)

    print(f"Epoch {i}: Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}")

print(f"Final Test Loss: {test_loss:4f}")

plt.plot(range(epochs), training_loss_graph, label='Training Loss')
plt.plot(range(epochs), validation_loss_graph, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.show()

# edit epochs to 50 and add [i] to the 2nd to last print statement for the funny