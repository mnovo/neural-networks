# Michael Novo

from __future__ import division
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp
from matplotlib.legend_handler import HandlerLine2D


# Make arrays (vectors) for each of the "true" values for letters

real_a = [-1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, +1]
real_e = [+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, +1, +1]
real_i = [-1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1]
real_o = [+1, +1, +1, +1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, +1, +1, +1, +1]
real_u = [+1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1, +1, +1, +1, +1]

training_xy = [[2,1],[3,2],[3,4],[4,4],[2,2],[4,2],[5,3],[4,3],[1,1],[4,2],[1,4],[4,3],[2,2],[4,2],[2,3],[4,3],[2,1],[4,1],[2,3],[4,3]]
toggle_xy_1 = [[4,1],[5,3],[2,4],[1,2],[2,4],[5,2],[4,3],[1,3],[4,4],[2,4],[2,2],[3,4],[1,3],[3,2],[5,4],[3,3],[5,3],[3,4],[1,4],[5,4]]
toggle_xy_2 = [[3,2],[1,3],[3,3],[2,1],[3,2],[3,3],[5,4],[1,2],[1,3],[2,2],[4,3],[1,4],[4,3],[4,3],[5,1],[2,4],[3,3],[2,2],[2,1],[1,3]]
toggle_xy_3 = [[5,2],[3,3],[1,3],[5,1],[3,4],[1,4],[5,2],[1,3],[1,1],[1,2],[2,4],[2,1],[1,2],[4,2],[3,3],[1,4],[3,2],[5,2],[5,4],[4,2]]

target_a = np.array([[-1, -1, 1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1], 
	    [-1, 1, -1], [-1, 1, -1], [-1, 1, -1], [-1, 1, -1], [-1, 1, -1],
	    [-1, 1, 1], [-1, 1, 1], [-1, 1, 1], [-1, 1, 1], [-1, 1, 1],
	    [1, -1, -1], [1, -1, -1], [1, -1, -1], [1, -1, -1], [1, -1, -1],    	
	    [1, -1, 1], [1, -1, 1], [1, -1, 1], [1, -1, 1], [1, -1, 1]])

        
# Function to convert (x,y) coordinates to corresponding index
def get_index(x, y):
    index = (y-1)*5 + (x-1)
    return index
    

# will create a 2d array containing 25 arrays within it
# Notes: Creates training and test sets
def create_feature_set(indices):
    # get real_a and copy it
    a_copy = copy.copy(real_a)
    # now put a in this 2d master array, append
    master = [[0 for x in range(20)] for x in range(25)] 
    # for loop to copy a into master
    for j in range(0,5):	
    	for i in range(len(real_a)):
		master[j][i] = real_a[i]
    # for loop to copy e into master
    for j in range(5,10):
	for i in range(len(real_e)):
		master[j][i] = real_e[i]
    # for loop to copy i into master
    for j in range(10,15):
	for i in range(len(real_i)):
		master[j][i] = real_i[i]
    # for loop to copy o into master
    for j in range(15,20):
	for i in range(len(real_o)):
		master[j][i] = real_o[i]
    # for loop to copy u into master
    for j in range(20,25):
	for i in range(len(real_u)):
		master[j][i] = real_u[i]
    # go through indices, and switch on corresponding indices
    for k in range(len(indices)):
	if k > 15:
	  # Modify the U's
	  master[k+5][get_index(indices[k][0], indices[k][1])] *= -1	
	elif k > 11:
	  # Modify the O's
	  master[k+4][get_index(indices[k][0], indices[k][1])] *= -1
	elif k > 7:
	  # Modify the I's
	  master[k+3][get_index(indices[k][0], indices[k][1])] *= -1
	elif k > 3:
	  # Modify the E's
	  master[k+2][get_index(indices[k][0], indices[k][1])] *= -1
	elif k >-1:
	  # Modify the A's
	  master[k+1][get_index(indices[k][0], indices[k][1])] *= -1

    return master


# Gets sum of product of v_weights with Xi
def hidden_layer(input_x, number_hidden_layers, v_weights):
 # Make an array for the Z_in
  z_in = np.array([float(0.00) for x in range(number_hidden_layers)])

  for j in range(len(z_in)): # Goes to 30
    z_in[j] += np.dot(input_x, v_weights[:len(v_weights)-1, j]) + v_weights[len(v_weights)-1][j]

  return z_in

# Evaluate the bipolar sigmoid
def bipolar_sigmoid(array):
  copy = np.array([float(0.00) for x in range(len(array))])
  for i in range(len(copy)):
    copy[i] = -1 + (2/(1 + np.exp(-array[i])))
  return copy

# Get sum, y_in, for each output element
def output_layer(input_z, number_output_layers, w_weights):
  # An array for y_in
  y_in = [float(0.00) for x in range(number_output_layers)]

  # For each output unit, do sum of weights and z_in, also add the bias terms
  for i in range(number_output_layers):
    if (number_output_layers == 1):
      y_in[i] += np.dot(input_z, np.transpose(w_weights[:len(w_weights)-1])) + w_weights[len(w_weights)-1]
    else:
      y_in[i] += np.dot(input_z, np.transpose(w_weights[:len(w_weights)-1, i])) + w_weights[len(w_weights)-1][i]

  return y_in

# Does backprop step, returns new v_weights and w_weights
def backprop(input_x, target, alpha, mu, v_weights, v_change_old, w_weights, w_change_old, z_in_j, z_j, y_in_k, y_k):

  # For plotting max weight change in each epoch
  max_weight_change = 0
  
  # First get the delta_k's
  delta_k = [float(0.00) for x in range(len(y_k))]
  for i in range(len(delta_k)):
    delta_k[i] = (target[i] - y_k[i])*0.5*(1+y_k[i])*(1-y_k[i])

  # Calc weight and bias correction; has as its last row the bias change
  w_change = np.array([[float(0.00) for x in range(len(y_k))] for x in range(len(z_j)+1)]) 
  
  for i in range(len(w_change)): # For each row (30)
    for j in range(len(w_change[0])): # For each column (3)
      if (i == len(w_change)-1):
        w_change[i][j] = alpha*delta_k[j]
	max_weight_change = max(max_weight_change, alpha*delta_k[j])
      else:
        w_change[i][j] = alpha*delta_k[j]*z_j[i]
	max_weight_change = max(max_weight_change, alpha*delta_k[j]*z_j[i])


  # Now get the delta_in_j's
  delta_in_j = [float(0.00) for y in range(len(z_j))]
  delta_j = [float(0.00) for y in range(len(z_j))] 

   	
  for i in range(len(w_weights)-1):
    for j in range(len(w_weights[0])):
      delta_in_j[i] += delta_k[j]*w_weights[i][j]
  # Get the delta_j
  for i in range(len(delta_j)):
    delta_j[i] = delta_in_j[i]*0.5*(1+z_j[i])*(1-z_j[i])	
  
 
  # V_weight correction
  # Create a 2d array, same size as v_weights, which includes the last bias row
  v_change = np.array([[float(0.00) for x in range(len(v_weights[0]))] for x in range(len(v_weights))])
  # do calc and to the Vij spot
  for i in range(len(v_change)):
    for j in range(len(v_change[0])):
      if (i == len(v_change)-1):
        v_change[i][j] = alpha*delta_j[j]
	max_weight_change = max(max_weight_change, alpha*delta_j[j])
      else:
	v_change[i][j] = alpha*delta_j[j]*input_x[i]
	max_weight_change = max(max_weight_change, alpha*delta_j[j]*input_x[i])
	

  # include a different calc for biases
  

  # Update v_weights
  # Add what was calculated above into the original v_weights
  for i in range(len(v_weights)):
    for j in range(len(v_weights[0])):
      v_weights[i][j] += v_change[i][j] + mu*v_change_old[i][j]


  # Update w_weights
  # Add what was calculated above into the original w_weights
  # be mindful of the bias terms
  for i in range(len(w_change)):
    for j in range(len(w_change[0])): 
      w_weights[i][j] += w_change[i][j] + mu*w_change_old[i][j]

  # DONE!
  return v_weights, v_change, w_weights, w_change, max_weight_change

# Returns the mean squared error; sum of error squared of each training pattern
# divided by the number of training patterns
def mean_squared_error(y_k, target):
  error = 0
  for i in range(len(y_k)):
    error += (y_k[i] - target[i])**2  
  return error/3

# Compare the y_k to target
# Returns 1 if correct, 0 if incorrect
def is_guess_correct(y_k, target):  
  for i in range(len(y_k)):
    if ((y_k[i]) >= 0):
      guess = 1 
    else:
      guess = -1
    if (guess != target[i]):
      return 0       	
  return 1    
 

# Plot function used in Perceptron and Adaline Functions
def plot(array, plt_title, x_title, y_title, class_name, learning_rate, tolerance):
  plt.figure()
  x_values = [(i+1) for i in range(len(array))]
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  
  if (plt_title[:3] == 'Ada'):
    plt_title = plt_title + ' alpha: ' + str(learning_rate) + ' tolerance: ' + str(tolerance)
  else:
    plt_title = plt_title + ' alpha: ' + str(learning_rate)
  plt.title(class_name + plt_title)  

  plt.axis('auto')
  plt.plot(x_values, array, 'ro')
  plt.savefig(plt_title + '.png')
  plt.show()

# Plots 5 weight arrays that are of size (number_of_patterns * number_of_epochs)
def plot_weights(weight1, weight2, weight3, weight4, weight5, plt_title, x_title, y_title, learning_rate):
  plt.figure()
  x_values = [(i+1) for i in range(len(weight1))]
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  
  plt.title(plt_title)
  line1, = plt.plot(x_values, weight1, 'r', label = 'Weight 1')
  line2, = plt.plot(x_values, weight2, 'b', label = 'Weight 2')
  line3, = plt.plot(x_values, weight3, 'g', label = 'Weight 3')
  line4, = plt.plot(x_values, weight4, 'k', label = 'Weight 4')
  line5, = plt.plot(x_values, weight5, 'm', label = 'Bias 1')
 
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

  plt.axis('auto')
  plt.savefig(plt_title + '.png')
  plt.show()


#############################################################################

def main():

 # Create our training set and test sets using the above fuctions
  training_set = create_feature_set(training_xy)
  test_set_1 = create_feature_set(toggle_xy_1)
  test_set_2 = create_feature_set(toggle_xy_2)
  test_set_3 = create_feature_set(toggle_xy_3)

  # Should be called hidden_units instead of hidden_layers, whoops!
  number_hidden_layers = 40
  number_output_units = 3

  # hidden_layers + 1, takes into account our desire to put the bias term at the end 
  v_weights = np.array([[random.uniform(-0.5, 0.5) for x in range(number_hidden_layers)] for x in range(len(training_set[0])+1)])

  # Nguyen-Widrow initialization  
  n = len(training_set)
  p = number_hidden_layers
  beta = 0.7*(np.power(p, 1/n))   
  
  for i in range(len(v_weights[0])): # Iterates through columns
    sq = np.square(v_weights[:len(v_weights)-1, i])
    v_j_old = np.power(np.sum(sq), 1/2)    
    for j in range(len(v_weights)): # Iterates through each row
      if (j == len(v_weights) -1): # Last row is bias, update with random beta range
        v_weights[j][i] = random.uniform(-beta, beta)
      else:
	v_weights[j][i] = (beta*v_weights[j][i])/v_j_old 


  v_change_old = np.array([[0 for x in range(number_hidden_layers)] for x in range(len(training_set[0])+1)])

  # Make random weights for w's
  w_weights = np.array([[random.uniform(-0.5, 0.5) for x in range(number_output_units)] for x in range(number_hidden_layers+1)])
  w_change_old = np.array([[0 for x in range(number_output_units)] for x in range(number_hidden_layers+1)])

  mse = 10
  mse_whole_set = 10
  epoch = 0
  mse_array = []
  max_weight_change_pattern = []
  max_weight_change_epoch = []
  weight1_evol_pattern = [] 
  weight2_evol_pattern = []
  weight3_evol_pattern = []
  weight4_evol_pattern = []
  weight5_evol_pattern = []
  mse_cutoff = 0.01
  alpha = 0.1
  mu = 0.9
  while (mse_whole_set > mse_cutoff):
    # Traing to get v_weights and w_weights
    epoch += 1	
    mse_whole_set = 0
    max_weight_change_pattern = []  # Reset this array for each epoch  
    for i in range(len(training_set)):

      # Get Z_in for each element   
      input_x = training_set[i] 
      target = target_a[i, :]
      z_in_j = hidden_layer(input_x, number_hidden_layers, v_weights)


      # Get z_j by doing bipolar sigmoid function
      z_j = bipolar_sigmoid(z_in_j)
  
      #  Get the y_in for the output layer
      y_in_k = output_layer(z_j, number_output_units, w_weights) 
      y_k = bipolar_sigmoid(y_in_k)


      mse = mean_squared_error(y_k, target)
      mse_whole_set += mse/(len(training_set))
      
  
      # Get the delta_k
      v_weights, v_change_old, w_weights, w_change_old, max_weight_change = backprop(input_x, 
							target, alpha, mu, v_weights, v_change_old, w_weights, w_change_old, z_in_j, z_j, y_in_k, y_k)   
      max_weight_change_pattern.append(max_weight_change)

      weight1_evol_pattern.append(w_weights[0][0])
      weight2_evol_pattern.append(w_weights[1][1])
      weight3_evol_pattern.append(w_weights[2][2])
      weight4_evol_pattern.append(w_weights[10][1])
      weight5_evol_pattern.append(w_weights[len(w_weights)-1][0])      

    mse_array.append(mse_whole_set)
    max_weight_change_epoch.append(max(max_weight_change_pattern))


  # Plot for Mean Squared Error
  plt_title = 'MLP NG MOM mse for each epoch ' + repr(number_hidden_layers) + ' hidden units'
  x_title = 'Epoch'
  y_title = 'mean squared error'
  class_name = ''
  learning_rate = repr(alpha) 
  tolerance = 0
  plot(mse_array, plt_title, x_title, y_title, class_name, learning_rate, tolerance)

  # Plot for Max Weight change Per Epoch
  plt_title = 'MLP NG MOM max weight change for each epoch ' + repr(number_hidden_layers) + ' hidden units'
  x_title = 'Epoch'
  y_title = 'max weight change'
  class_name = ''
  learning_rate = repr(alpha) 
  tolerance = 0
  plot(max_weight_change_epoch, plt_title, x_title, y_title, class_name, learning_rate, tolerance)

  # Plot for weights and biases
  plt_title = 'MLP NG MOM Weight evolution per pattern ' + repr(number_hidden_layers) + ' hidden units'
  x_title = 'pattern'
  y_title = 'weight value'
  class_name = ''
  learning_rate = repr(alpha) 
  tolerance = 0
  plot_weights(weight1_evol_pattern, weight2_evol_pattern, weight3_evol_pattern, weight4_evol_pattern,
                   weight5_evol_pattern, plt_title, x_title, y_title, learning_rate)

  # Save the v and w weights
  np.savetxt('v_weights.csv', v_weights)
  np.savetxt('w_weights.csv', w_weights)
 


  # Destroy 20% of v_weights to test fault tolerance  
  percent_destroy = 0.4
  break_num = int( percent_destroy * (len(v_weights)-1) * len(v_weights[0]) )
  for i in range (break_num):
    while True:
      row_destroy = random.randint(0, len(v_weights)-2)
      col_destroy = random.randint(0, len(v_weights[0])-1)     
      if (v_weights[row_destroy][col_destroy] != 0):
        v_weights[row_destroy][col_destroy] = 0
  	break  
  
  # Destroy 20% of w_weights to test fault tolerance
  break_num = int (percent_destroy * (len(w_weights)-1) * len(w_weights[0]))
  for i in range (break_num):
    while True:
      row_destroy = random.randint(0, len(w_weights)-2)
      col_destroy = random.randint(0, len(w_weights[0])-1)
      if (w_weights[row_destroy][col_destroy] != 0):
        w_weights[row_destroy][col_destroy] = 0
        break 

  for j in range(3):
    # Run test to classify the testing sets
    correct_predictions = 0
    if j == 0:
      test_set = test_set_1
    if j == 1:
      test_set = test_set_2
    if j == 2:
      test_set = test_set_3
    for i in range(len(test_set)):
      input_x = test_set[i]
      target = target_a[i, :]
      
      z_in_j = hidden_layer(input_x, number_hidden_layers, v_weights)
      z_j = bipolar_sigmoid(z_in_j)
        
      y_in_k = output_layer(z_j, number_output_units, w_weights) 
      y_k = bipolar_sigmoid(y_in_k)  
      correct_predictions += is_guess_correct(y_k, target)
     
    print 'Hits (out of ' + repr(len(test_set_1)) + '): '  + repr(correct_predictions)    
    print 'Misses (out of ' + repr(len(test_set_1)) + '): ' + repr(len(test_set_1) - correct_predictions)
    print 'Hit Ratio: ' + repr(float(correct_predictions)/(len(test_set_1)))

##################################################################################
# Use function from above to train Adaline for class A

 

if __name__ == "__main__":
	main()

###############################################################################


