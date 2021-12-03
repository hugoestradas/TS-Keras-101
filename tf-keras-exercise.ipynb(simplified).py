# Part 1 - Imports

# I'm going to use the descriptively named “make_blobs” method from sklearn to create the data.
from sklearn.datasets import make_blobs
# And I'm going to use NumPy to manipulate the data
import numpy as np
# And matplotlib to plot the data
import matplotlib.pyplot as plt
# Then, I will import “os” to set the environment so I don't see the TensorFlow warning messages:
os.environ["TF_CPP_MIN_LOG_LEVEL"]=2

# Part 2 - Helper Functions

# The first is called plot_data, and as you probably 
# guessed it plots the clusters of data
def plot_data(pl, X, y):
    # plot class where y==0
    pl.plot(X[y==0, 0], X[y==0,1], 'ob', alpha=0.5)
    # plot class where y==1
    pl.plot(X[y==1, 0], X[y==1,1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl

# The second function is called plot_decision_boundary and it plots
# a separation, I will learn between the two clusters of data.
# This function is a variant of one that is provided as an example
# in scikit-learn, this one plots a contour of the boundary rather than
# a single line, so I can see the confidence of the prediction.
def plot_decision_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    # make prediction with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_data(plt, X, y)

    return plt

# Part 3 - Problem's specific code

# I call make_blobs to create data from the number of data points
# defined in the contour parameter.
# The center=2 parameter tells the functino to return two clusters
# of data, one cluster for class 1 and another for class 0.
# The x array contains the coordinates of each data point 
# and the x array contain the class 0 or 1 of the data point.
X, y = make_blobs(n_samples=1000,\
    centers=2, random_state=42)

# Let's plot these distributions to make sure they look reasonable.
p1 = plot_data(plt, X, y)
p1.show()

# Description of the data.
# Here are our clusters, they look nice and easily separable.
# So I don't need much of a network to figure out how to define
# the boundary between the two.
# In general, any data distribution that is lineraly separable, 
# that is that you can put a plane between, can be done without
# needing any hidden layers in a neural network.
# So I will define our initial neural network withot any hidden layers.
# and I will add the hidden layers later if I need them to separate
# the data from each cluster.

# Now I will split the data into training and test datasets.

# I will train the model with the trainint data and then evaluate the
# performance of the model with the testing data.
from sklearn.model_selection import train_test_split

# By setting the test size to 0.3, I am putting 30% of the data in the
# X_test and y_test arrays.
# And the remainint 70% goes in the X_train and y_train arrays.
# I also set the initializer of the random_state to ensure we can 
# produce the same split if we run the code multiple times.
X_train, X_test, \
y_train, y_test = \
    train_test_split(\
        X, y, test_size=0.3,\
            random_state=42)

# From Keras, I will import the sequential model and dense layer.
# In the dense layer, every neuron is connected to every neuron in 
# the following layer, or to the ouput, if there's not a following
# layer.
from keras.models import Sequential
from keras.layers import Dense

# Then I import the Adam optimizer, which will perform 
# back propagation to adjust the weights and biases to minimize the
# error during training.
from tensorflow.keras.optimizers import Adam

# Now I'm ready to do trainint and evaluation.

# The general pattern for the sequential model is:
# create_model >> which creates the sequential model
# add_layers >> in the order from input to output
# compile_model
# train_model >> with the training data
# and.. evaluate_model >> measures the performance of the model against the testing/validation data.

# Let's start by definint a sequential model.
# This is a simple model in which each layer is inserted at the end of
# the network, and gets the input from the previous layers or from the data
# passed in the case of the first layer.
model = Sequential()

# Now I need to add our first, and in this example, only layer.
# This is a dense layer that I will train to divide the two classes
# of data.
# I specify that the dense layer contains only one neuron, which
# will be whether the data belongs to class 0 or class 1.
# I am expecting two values, the x and y positions, for each data
# so I can set the input shape to "2,": a one-dimensional array of two elements.
model.add(Dense(\
         1, input_shape=(2,),\
         activation="sigmoid"))

# I define the model's learning process by calling the compile method.
# I specify the Adam optimizer to minimize the loss, which if how often
# the model incorrectly predicts the class.
# And the binary_crossentropy function is used to calculate the loss, 
# and that the accuracy is a metric we want to optimize.
model.compile(Adam(\
                  learning_rate=0.05),\
                  'binary_crossentropy',\
                  metrics=['accuracy'])

# With the model defined, I use the fit method to adjust the weight and
# bias in the model to minimize the loss.
# I do this, by running the training set through the fit method a 
# specified number of times.
# Each run through the data is called an epoch.
# Therefore with the epoch set to 100, I am making a 100 runs though 
# the training data.
# And on each run the optimizer will adjust the network to minimize
# the loss and increase the accuracy.
model.fit(X_train, y_train, epochs=10, verbose=1)

# Once the model is trained, I use the evaluate method to evaluate 
# how well my model predicts the class of the locations and the test set 
# of data.
eval_result = model.evaluate(X_test, y_test)

# I print out the results of this evaluation as numeric values for 
# loss and accuracy.
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])

# And just to make it easier to understand, I will plot the decision
# boundary of the model I've learned.
plot_decision_boundary(model, X, y).show()
