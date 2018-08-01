#Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import math
import scipy
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from PIL import Image
from scipy import ndimage

%matplotlib inline

y_hat = tf.constant(36, name = "y_hat")
y = tf.constant(39, name = "y")

loss = tf.Variable((y - y_hat)**2, name = "loss")

init = tf.Session() as session:
  session.run(init)
  print(session.run(loss))
  
#Initialize, Create a session and run the operations inside the session
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
sess = tf.Session()
print(sess.run(c))

#placeholders
x = tf.placeholder(tf.int64, name = "x")
print(sess.run(9 * x + x, feed_dict = {x : 7}))
sess.close()

#linear function
def linear_function():
  X = tf.constant(np.random.randn(10, 1), name = "X")
  W = tf.constant(np. random.randn(5, 10), name = "W")
  b = tf.constant(np.random.randn(5, 1), name = "b")
  Y = tf.add(tf.matmul(W, X), b)
  
  sess = tf.Session()
  result = sess.run(Y)
  
  sess.close()
  
  return result
  
print("result = " + str(linear_function()))

#Session
#1 - sess = tf.Session()    sess.run()    sess.close()
#2 - with tf.Session() as sess:   sess.run()

def sigmoid(z):
  x = tf.placeholder(tf.float32, name = "x")
  
  sigmoid = tf.sigmoid(x)
  
  with tf.Session() as sess:
    result = sess.run(sigmoid, feed_dict = {x : z})
    
  return result
  
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(10) = " + str(sigmoid(10)))

def cost(logits, lables):
  z = tf.placeholder(tf.float32, name = "z")
  y = tf.placeholder(tf.float32, name = "y")
  
  cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
  
  sess = tf.Session()
  
  cost = sess.run(cost, feed_dict = {z: logits, y: labels})
  
  sess.close()
  
  return cost
  
logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = cost(logits, np.array([0, 0, 1, 1]))
print ("cost = " + str(cost))

def one_hot_matrix(labels, C):
  C = tf.constant(C, name = "C")
  
  one_hot_matrix = tf.one_hot(indices = labels, depth = C, axis = 0)
  
  sess = tf.Session()
  
  one_hot = sess.run(one_hot_matrix)
  
  sess.close()
  
  return one_hot

labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C=4)
print ("one_hot = " + str(one_hot))

def ones(shape):
  ones = tf.ones(shape)
  
  sess = tf.Session()
  
  ones = sess.run(ones)
  
  sess.close()
  
  return ones

print("ones = " + str(ones([7])))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

#Create placeholders
def create_placeholders(n_x, n_y):
  X = tf.placeholder(tf.float32, [n_x, None], name = "X")
  Y = tf.placeholder(tf.float32, [n_y, None], name = "Y")
  
  return X, Y
  
X, Y = create_placeholders(12288, 6)
print("X = " + str(X))
print("Y = " + str(Y))

#Initialize parameters
def initialize_parameters():
  W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("b1", [25, 1], initializer = tf.contrib.layers.xavier_initializer())
  W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("b2", [12, 1], initializer = tf.contrib.layers.xavier_initializer())
  W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer())
  b3 = tf.get_variable("b3", [6, 1], initializer = tf.contrib.layers.xavier_initializer())
  
  parameters = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}
                
  return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

#forward propagation
def forward_propagation(X, parameters):
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  W3 = parameters['W3']
  b3 = parameters['b3']
  
  Z1 = tf.add(tf.matmul(W1, X), b1)
  A1 = tf.nn.relu(Z1)
  Z2 = tf.add(tf.matmul(W2, A1), b2)
  A2 = tf.nn.relu(Z2)
  Z3 = tf.add(tf.matmul(W3, A2), b3)
  
  return Z3
  
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))

#cost
def compute_cost(Z3, Y):
  logits = tf.transpose(Z3)
  labels = tf.transpose(Y)
  
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  
  return cost
  
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))

#Training and Predicting in TensorFlow
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1501, minibatch_size = 32, print_cost = True):
  ops.reset_default_graph()
  (n_x, m) = X_train.shape
  n_y = Y_train.shape[0]
  costs = []
  
  X, Y = create_placeholders(n_x, n_y)
  
  parameters = initialize_parameters()
  
  Z3 = forward_propagation(X, parameters)
  
  cost = compute_cost(Z3, Y)
  
  optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
  
  init = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
      epoch_cost = 0.
      num_minibatches = int(m / minibatch_size)
      
      minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
      
      for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        
        _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
        
        epoch_cost += minibatch_cost / num_minibatches
        
      if print_cost == True and epoch % 100 == 0:
        print("cost after epoch %i: %f" % (epoch, epoch_cost))
      if print_cost == True and epoch % 5 == 0:
        costs.append(epoch_cost)
  
    plt.plot(np.squeeze(costs))
    plt.xlabel("iterations (per tens)")
    plt.ylabel("cost")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
  
    parameters = sess.run(parameters)
    print("Parameters have been trained!")
  
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print("Train Accuracy: ", accuracy.eval({X: X_train, Y: Y_train}))
    print("Test Accuracy: ", accuracy.eval({X: X_test, Y: Y_test}))
    
    return parameters

parameters = model(X_train, Y_train, X_test, Y_test)

#Testing with an Image
my_image = "image.jpg"

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten = False)
my_image = scipy.misc.imresize(image, size = (64, 64)).reshape((1, 64 * 64 * 3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
