# Import tensorflow
import tensorflow as tf

# Importing the hand written digit data set
# input_data class is a standard python class which 
#      - Downloads the dataset 
#      - Splits into training and testing data 
#      - and formats the data

# Forked from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py

import input_data
mnist = input_data.read_data_sets("/Users/Raj/Root/Code/TF/data/", one_hot =True)


# Setting the  parameters

# learning_rate Defines how fast we want to update our weights
#    If the learning rate is too big our model might skip the solution
#    and if it is too small we might need too many iterations to converge on the optimum result
learning_rate = 0.01

# setting the number of iterations
training_iteration = 30

batch_size = 100
display_step = 2

# TF graph input
# We first create two placeholder operations, placeholder is the variable in which we will assign the data later
#     Its never initialized and doesn't contain any data 
# We define the type and shape of the data as the parameters

# Input image is x and will be represented by a 2D tensor of numbers - [None, 784]
#    784 is the dimensionality of the single flattend MNIST image
# Finding an image means converting a 2D array into an 1D array, by unstacking the rows and lining them up
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784

# y is The output class and will consist of a 2D tensor as well, ...
#    ...where each row is a one_hot 10 dimensional vector, showing which digit class the correspoding ...
#    ...MNIST image belongs to
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes


# Start building the Model

# Define model weights and Biases
# weights are the probabilities that affect how data flows in the graph and they get updated continuously ...
#    ...during the training, so that our results get closer and closer to the right solution
W = tf.Variable(tf.zeros([784, 10]))
# The Bias lets us shift our regression line to better fit the data
b = tf.Variable(tf.zeros([10])) 


# Now we create the Name Scope
# Scopes helps us organize the nodes in the graph visualizer called Tensor Board

# In this case we create 3 scopes
# In the first scope we Implement our model, Logistic Regression
#    By matrix multiplying)(tf.matmul) the input images x by the weight matrix w and adding the Bias b
with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


# In the second scope we will create our cost function which will help us minimize...
#    ...our error during the training.
# We use the cross entropy function 
# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function during training to visualize it later
    tf.scalar_summary("cost_function", cost_function)

# Our last scope is called train and it will create our optimization function that ...
#    ...makes our model improve during training
# And we use the popular Gradient descent algorith which takes our learning_rate as a parameter...
#    ...for pacing and our cost_function as a parameter to help minimize the error
with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


# Now that we have our graph built we will initialize all of our variables... 
# Initializing the variables
init = tf.initialize_all_variables()

# ...then we will merge all of our summaries into a single operator
# Merge all summaries into a single operator
merged_summary_op = tf.merge_all_summaries()



# Now we are ready to launch our graph by initializing a session which lets us execute our data flow graph

# LAUNCH THE GRAPH
with tf.Session() as sess:
    sess.run(init)

    # We will then set our summary writer folder location which will later load data from ...
    #    ...to visualize in tensor board
    
    # set the logs writer to the folder 
    summary_writer = tf.train.SummaryWriter('/Users/Raj/Root/Code/TF/logs/', graph=sess.graph)

    
    # TRAINING CYCLE

    # lets set our for loop specified number of iterations(training_iteration) 
    for iteration in range(training_iteration):
        # initialize the average cost which we will print out every so often to make sure...
        #    ...our model is improving during training 
        avg_cost = 0.
        # Now we will compute our batch size...
        total_batch = int(mnist.train.num_examples/batch_size)
        # ...and start training over each example in our training data
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data in the Gradeint descent algorithm for Back propogation
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration via the summary writer 
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# Visualize the graph in tensor board
# $ tensorboard --logdir = /Users/Raj/Root/Code/TF/logs/

# ref: https://goo.gl/6pFKQt

