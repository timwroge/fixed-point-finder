import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn metrics import mean_squared_error
minimum = 0
maximum = 60
data_points = np.linspace(minimum, maximum, (maximum - minimum)*10)
dataset = np.sin(data_points)
dataset = dataset.reshape(-1,1) # necessary for scaler fit_transform function
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
n_steps = 100
n_iterations = 10000
n_inputs = 1 # one input per time step
n_neurons = 120 # one hidden layer
n_outputs = 1 # output layer
learning_rate = 0.0001
dataset = dataset.reshape(-1,) # reshape it back
dataX, dataY = create_training_dataset(dataset, n_steps, n_outputs)
def create_training_dataset(dataset, n_steps, n_outputs):
    dataX, dataY = [], []
    for i in range(500):
        x = dataset[i]
        y = dataset[i+1]
        dataX.append(x)
        dataY.append(y)
    dataX, dataY =  np.array(dataX), np.array(dataY)
    dataX = np.reshape(dataX, (-1, n_steps, n_outputs))
    dataY = np.reshape(dataY, (-1, n_steps, n_outputs))
    return dataX, dataY

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
# initialize all variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = dataX, dataY
        # prediction dimension [batch_size x t_steps x n_inputs]
        _, prediction =sess.run((training_op, outputs), feed_dict={X: X_batch, y: y_batch})
        if iteration % 20 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE", mse)
            # roll out prediction dimension into a single dimension
            prediction = np.reshape(prediction, (-1,))
            plt.plot(prediction)
            plt.title('prediction over training data')
            plt.show()

            # simulate the prediction for some time steps
            #sequence = [0.]*n_steps
            num_batches = X_batch.shape[0]
            sequence = X_batch[num_batches
1,:,:].reshape(-1).tolist()
            prediction_iter = 100
            for iteration in range(prediction_iter):
                X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
                y_pred = sess.run(outputs, feed_dict={X: X_batch})
                sequence.append(y_pred[0, -1, 0])
            plt.plot(sequence[-prediction_iter:])
            plt.title('prediction')
            plt.show()
