# Michael Novo

from __future__ import division
import copy
import numpy as np
import random
import matplotlib.pyplot as plt


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

target_a = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
target_e = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
target_i = [-1, -1, -1, -1, -1 ,-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
target_o = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ,-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
target_u = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ,-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

        
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


# Function that implements hebb learning algorithm seen in text and
# in class notes
def hebb_learning(input_x, input_y):
    # Initialize weight vector of size 25 and bias to 0
    weights = [0 for x in range(20)]
    bias = 0

    # Initilize training; update weights and biases accordingly
    for i in range(len(input_x)):
	bias = bias + input_y[i]
    	for j in range (len(input_x[0])):
		weights[j] = weights[j] + input_x[i][j]*input_y[i]
    # Add the bias to the last element of the weight vector   
    weights.append(bias)		
    return weights

# Function to implement the perceptron algorithm
# Returns the set of weights with bias appended to end of array
def perceptron(input_x, input_y, learning_rate, class_name):
    # Initialize weights and bias
    weights = [0 for x in range(20)]
    bias = 0 
    guess = 0
    weights_changed = 1
    epochs = 0
    largest_weight_change = 0
    max_weight_change_list = []
    weight1_list = []
    weight2_list = []
    weight3_list = []
    weight4_list = []
    weight5_list = []
    bias_list = []
   
    while (weights_changed == 1):
     weights_changed = -1
     largest_weight_change = 0	
     epochs = epochs + 1
     for i in range(len(input_x)):	
      y_in = bias + np.dot(input_x[i], weights[:20])
      if y_in >= 0:
       guess = 1
      else:
       guess = -1
      if guess != input_y[i]:	
       # Update weights and bias, with learning rate
       bias = bias + learning_rate*input_y[i]
       bias_list.append(bias)
       weights_changed = 1
       for j in range(len(weights)):
        weights[j] = weights[j] + learning_rate*input_y[i]*input_x[i][j]
	weight_change = learning_rate*input_y[i]*input_x[i][j]
	if (j == 0):
          weight1_list.append(weights[j]) 
        if (j == 1):
	  weight2_list.append(weights[j])
        if (j == 2):
	  weight3_list.append(weights[j])
        if (j == 3):
	  weight4_list.append(weights[j])
        if (j == 4):
	  weight5_list.append(weights[j]) 
        if (weight_change > largest_weight_change):
          largest_weight_change = weight_change    	              
      else: # Keep old values
       weights = weights
       bias = bias
     max_weight_change_list.append(largest_weight_change)
  
    # Required by plot function below
    tolerance = 0

    # Plot the evolution of max weight change per epoch
    plot(max_weight_change_list, 'Perceptron Max Weight Change per Epoch', 'Epochs', 'Max Weight Change', class_name, learning_rate, tolerance)
    # Plot the evolution of weight 1
    plot(weight1_list, 'Perceptron Evolution of Weight 1 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
    # Plot the evolution of weight 2
    plot(weight2_list, 'Perceptron Evolution of Weight 2 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
    # Plot the evolution of weight 3
    plot(weight3_list, 'Perceptron Evolution of Weight 3 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
    # Plot the evolution of weight 4
    plot(weight4_list, 'Perceptron Evolution of Weight 4 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
    # Plot the evolution of weight 5
    plot(weight5_list, 'Perceptron Evolution of Weight 5 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
    # Plot the evolution of bias
    plot(bias_list, 'Perceptron Evolution of Bias', 'Pattern Presentations', 'Bias', class_name, learning_rate, tolerance)
    # Plot the final weights 
    plot(weights[:20], 'Final Perceptron Weights', 'Weight Index', 'Weight Values', class_name, learning_rate, tolerance) 
     

     # Add the bias value to the end of the weight vector
    weights.append(bias)
    return weights	 

# Implementation of the adaline algorithm
def adaline(input_x, input_y, learning_rate, tolerance, class_name):
  # Initialize the weights and bias
  weights = [round(random.uniform(-0.5, 0.5), 2) for x in range(20)]
  bias = 0
  largest_weight_change = 100
  weight_change = 0
  epochs = 0
  max_weight_change_list = []
  weight1_list = [weights[0]]
  weight2_list = [weights[1]]
  weight3_list = [weights[2]]
  weight4_list = [weights[3]]
  weight5_list = [weights[4]]
  bias_list = [bias]
  
  while (largest_weight_change > tolerance ):    
    epochs = epochs + 1
    largest_weight_change = 0 
    for i in range(len(input_x)):	
     weight_change = 0	
     y_in = bias + np.dot(input_x[i], weights[:20])
      # Update weights and bias, with learning rate
     bias = bias + learning_rate*(input_y[i]-y_in)
     bias_list.append(bias)
     for j in range(len(weights)):      
      weights[j] = weights[j] + learning_rate*(input_y[i]-y_in)*input_x[i][j]	
      if (j == 0):
	weight1_list.append(weights[j])
      if (j == 1):
	weight2_list.append(weights[j])
      if (j == 2):
	weight3_list.append(weights[j])
      if (j == 3):
	weight4_list.append(weights[j])
      if (j == 4):
	weight5_list.append(weights[j])  
      weight_change = learning_rate*(input_y[i]-y_in)*input_x[i][j]
      if (weight_change > largest_weight_change):
       largest_weight_change = weight_change       	
       
    max_weight_change_list.append(largest_weight_change)

 
  # Plot the evolution of max weight change per epoch
  plot(max_weight_change_list, 'Adaline Max Weight Change per Epoch', 'Epochs', 'Max Weight Change', class_name, learning_rate, tolerance)
  # Plot the evolution of weight 1
  plot(weight1_list, 'Adaline Evolution of Weight 1 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
  # Plot the evolution of weight 2
  plot(weight2_list, 'Adaline Evolution of Weight 2 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
  # Plot the evolution of weight 3
  plot(weight3_list, 'Adaline Evolution of Weight 3 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
  # Plot the evolution of weight 4
  plot(weight4_list, 'Adaline Evolution of Weight 4 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
  # Plot the evolution of weight 5
  plot(weight5_list, 'Adaline Evolution of Weight 5 per Pattern', 'Pattern Presentations', 'Weights', class_name, learning_rate, tolerance)
  # Plot the evolution of bias
  plot(bias_list, 'Adaline Evolution of Bias', 'Pattern Presentations', 'Bias', class_name, learning_rate, tolerance)

  # Plot the final weights 
  plot(weights[:20], 'Final Adaline Weights', 'Weight Index', 'Weight Values', class_name, learning_rate, tolerance) 

  # Add the bias value to the end of the weight vector
  weights.append(bias)
  return weights

# Plot function used in Perceptron and Adaline Functions
def plot(array, plt_title, x_title, y_title, class_name, learning_rate, tolerance):
  x_values = [(i+1) for i in range(len(array))]
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  
  if (plt_title[:3] == 'Ada'):
    plt_title = plt_title + ' alpha: ' + str(learning_rate) + ' tolerance: ' + str(tolerance)
  if (plt_title[:3] == 'Per'):
    plt_title = plt_title + ' alpha: ' + str(learning_rate)
  plt.title(class_name + plt_title)  
  plot_min = 0
  plot_max = 0
  if (min(array) == 0):
    plot_min = -0.05
  if (min(array) > 0):
    plot_min = min(array) - 0.1*(min(array))
  else:
    plot_min = min(array) + 0.1*(min(array))

  if (max(array) == 0):
    plot_max = -0.05
  if (max(array) > 0):
    plot_max = max(array) + 1.1*max(array)
  else:
    plot_max = max(array) - 1.1*max(array)
 
  plt.axis([0, len(array)+1, plot_min, plot_max ])
  plt.plot(x_values, array, 'ro')
  plt.show()

# Note: weights should include bias in last element of vector
def test(weights, targets, test_set):
    num_incorrect = 0
    guess_vector = [0 for x in range(25)]
    for i in range(len(test_set)):
         dot_prod = np.dot(test_set[i], weights[:20])
	 net = dot_prod + weights[20]
	 guess = 0
	 if net >= 0:
	   guess = 1
	 else:
	   guess = -1

	 guess_vector[i] = guess

	 if guess != targets[i]:
	   num_incorrect += 1

    return guess_vector, num_incorrect	 

# Prints results of weights, bias, target, test class 1/2/3, number wrong of each 
# Note: bias value is last element of the weight vector
def print_results(alg_name, weights, targets, learning_rate, tolerance, class_test_1, wrong_test_1, class_test_2, 
 						wrong_test_2, class_test_3, wrong_test_3):
    print '--------------------------------------------------------'
    print 'Results for ' + alg_name
    print 'Weights: ',
    print weights[:20]
    print 'Bias: ',
    print weights[20]	
    if (alg_name[:3] == 'PER' or alg_name[:3] == 'ADA'):
	print 'alpha: ',
        print learning_rate
    if (alg_name[:3] == 'ADA'):
	print 'tolerance: ',
	print tolerance
	print 'Stopping criteria: Largest weight change'
    print 'Targets | Class Test 1 | Class Test 2 | Class Test 3'
    for i in range(len(targets)):
      print ('%5d %10d %13d %13d' % (targets[i], class_test_1[i], class_test_2[i], class_test_3[i]))
    
    print 'Misses Test Set 1: ',
    print wrong_test_1
    print 'Hits Test Set 1: ',
    print len(targets) - wrong_test_1
    print 'Hit Ratio: ',
    print (len(targets) - wrong_test_1)/(len(targets))

    print 'Misses Test Set 2: ',
    print wrong_test_2
    print 'Hits Test Set 2: ',
    print len(targets) - wrong_test_2
    print 'Hit Ratio: ',
    print (len(targets) - wrong_test_2)/(len(targets))

    print 'Misses Test Set 3: ',
    print wrong_test_3
    print 'Hits Test Set 3: ',
    print len(targets) - wrong_test_3
    print 'Hit Ratio: ',
    print (len(targets) - wrong_test_3)/(len(targets))
		
    print '--------------------------------------------------------'	

#############################################################################

def main():

 # Create our training set and test sets using the above fuctions
 training_set = create_feature_set(training_xy)
 test_set_1 = create_feature_set(toggle_xy_1)
 test_set_2 = create_feature_set(toggle_xy_2)
 test_set_3 = create_feature_set(toggle_xy_3)

####################################################################################
 # Use function from above to train HEBB LEARNER for class A
 hebb_weights_a = hebb_learning(training_set, target_a)

 # Test HEBB LEARNER trained above; make a function for this
 guess_test_1, incorrect_test_1 = test(hebb_weights_a, target_a, test_set_1)
 guess_test_2, incorrect_test_2 = test(hebb_weights_a, target_a, test_set_2) 
 guess_test_3, incorrect_test_3 = test(hebb_weights_a, target_a, test_set_3)

 # Tolerance only set for print_results function
 tolerance = 0
 learning_rate = 0
 print_results('HEBB Learning "A"', hebb_weights_a, target_a, learning_rate, tolerance, guess_test_1, incorrect_test_1, 
	guess_test_2, incorrect_test_2, guess_test_3, incorrect_test_3) 

###############################################################################
###############################################################################
 # 1st attempt - Use function from above to train PERCEPTRON for class A
 # 'Poor' attempt, do not show results 
 
 learning_rate_a = 1
 perceptron_weights_a = perceptron(training_set, target_a, learning_rate_a, 'A class ')
 
###############################################################################
 # 2nd attempt - Use function from above to train PERCEPTRON for class A
 # 'Poor' attempt, do not show results 

 learning_rate_a = 10
 perceptron_weights_a = perceptron(training_set, target_a, learning_rate_a, 'A class ')

##############################################################################
 # 3rd attempt - Use function from above to train PERCEPTRON for class A
 # This is the best training attempt and will be used on testing set
 
 learning_rate_a = 0.1
 perceptron_weights_a = perceptron(training_set, target_a, learning_rate_a, 'A class ')

 guess_test_percep_1_a, incorrect_test_percep_1_a = test(perceptron_weights_a, target_a, test_set_1)
 guess_test_percep_2_a, incorrect_test_percep_2_a = test(perceptron_weights_a, target_a, test_set_2)
 guess_test_percep_3_a, incorrect_test_percep_3_a = test(perceptron_weights_a, target_a, test_set_3)

 # Tolerance only set for print_results function
 tolerance = 0
 print_results('PERCEPTRON Learning "A"', perceptron_weights_a, target_a, learning_rate_a, tolerance, guess_test_percep_1_a, incorrect_test_percep_1_a, guess_test_percep_2_a, incorrect_test_percep_2_a, guess_test_percep_3_a, incorrect_test_percep_3_a)

#####################################################################################
#####################################################################################
 # 1st attempt - ADALINE to train A classifier
 # 'Poor' attempt, do not show results 
 learning_rate = 0.05
 tolerance = .01
 ada_weights_a = adaline(training_set, target_a, learning_rate, tolerance, 'A class ')
 
#####################################################################################
 # 2nd attempt - ADALINE to train A classifier
 # 'Poor' attempt, do not show results 
 learning_rate = 0.001
 tolerance = .1
 ada_weights_a = adaline(training_set, target_a, learning_rate, tolerance, 'A class ')

#####################################################################################
 # 3rd attempt - ADALINE to train A classifier
 # This is the best training attempt and will be used on testing set - and results shown
 learning_rate = 0.010
 tolerance = 0.010
 ada_weights_a = adaline(training_set, target_a, learning_rate, tolerance, 'A class ')
 
 guess_test_ada_1_a, incorrect_test_ada_1_a = test(ada_weights_a, target_a, test_set_1)
 guess_test_ada_2_a, incorrect_test_ada_2_a = test(ada_weights_a, target_a, test_set_2)
 guess_test_ada_3_a, incorrect_test_ada_3_a = test(ada_weights_a, target_a, test_set_3)

 print_results('ADALINE Learning "A"', ada_weights_a, target_a, learning_rate, tolerance, guess_test_ada_1_a, incorrect_test_ada_1_a,
  guess_test_ada_2_a, incorrect_test_ada_2_a, guess_test_ada_3_a, incorrect_test_ada_3_a)

####################################################################################
####################################################################################
####################################################################################
 # Use function from above to train HEBB LEARNER for class E
 hebb_weights_e = hebb_learning(training_set, target_e)

 # Test HEBB LEARNER trained above; make a function for this
 guess_test_1_e, incorrect_test_1_e = test(hebb_weights_e, target_e, test_set_1)
 guess_test_2_e, incorrect_test_2_e = test(hebb_weights_e, target_e, test_set_2) 
 guess_test_3_e, incorrect_test_3_e = test(hebb_weights_e, target_e, test_set_3)

 # Tolerance only set for print_results function
 tolerance = 0
 learning_rate = 0
 print_results('HEBB Learning "E"', hebb_weights_e, target_e, learning_rate, tolerance, guess_test_1_e, incorrect_test_1_e, 
	guess_test_2_e, incorrect_test_2_e, guess_test_3_e, incorrect_test_3_e) 

##################################################################################
##################################################################################
 # 1st attempt - Use function from above to train PERCEPTRON for class E
 # 'Poor' attempt, do not show results 
 
 learning_rate_e = 10
 perceptron_weights_e = perceptron(training_set, target_e, learning_rate_e, 'E class ')
 
##################################################################################
 # 2nd attempt - Use function from above to train PERCEPTRON for class E
 # 'Poor' attempt, do not show results 
 
 learning_rate_e = 1
 perceptron_weights_e = perceptron(training_set, target_e, learning_rate_e, 'E class ')

##################################################################################
 # 3rd attempt - Use function from above to train PERCEPTRON for class E
 # This is the best training attempt and will be used on testing set
 
 learning_rate_e = 0.1
 perceptron_weights_e = perceptron(training_set, target_e, learning_rate_e, 'E class ')
 
 guess_test_percep_1_e, incorrect_test_percep_1_e = test(perceptron_weights_e, target_e, test_set_1)
 guess_test_percep_2_e, incorrect_test_percep_2_e = test(perceptron_weights_e, target_e, test_set_2)
 guess_test_percep_3_e, incorrect_test_percep_3_e = test(perceptron_weights_e, target_e, test_set_3)
 # Tolerance only set for print_results function
 tolerance = 0
 print_results('PERCEPTRON Learning "E"', perceptron_weights_e, target_e, learning_rate_e, tolerance, guess_test_percep_1_e, incorrect_test_percep_1_e, guess_test_percep_2_e, incorrect_test_percep_2_e, guess_test_percep_3_e, incorrect_test_percep_3_e)


#####################################################################################
#####################################################################################
 # 1st attempt - ADALINE to train E classifier
 # 'Poor' attempt, do not show results 

 learning_rate = .05
 tolerance = .05
 ada_weights_e = adaline(training_set, target_e, learning_rate, tolerance, 'E class ')

########################################################################################
 # 2nd attempt - ADALINE to train E classifier
 # 'Poor' attempt, do not show results 

 learning_rate = .03
 tolerance = .03
 ada_weights_e = adaline(training_set, target_e, learning_rate, tolerance, 'E class ')

########################################################################################
 # 3rd attempt - ADALINE to train E classifier
 # This is the best training attempt and will be used on testing set

 learning_rate = .01
 tolerance = .01
 ada_weights_e = adaline(training_set, target_e, learning_rate, tolerance, 'E class ')
 
 guess_test_ada_1_e, incorrect_test_ada_1_e = test(ada_weights_e, target_e, test_set_1)
 guess_test_ada_2_e, incorrect_test_ada_2_e = test(ada_weights_e, target_e, test_set_2)
 guess_test_ada_3_e, incorrect_test_ada_3_e = test(ada_weights_e, target_e, test_set_3)

 print_results('ADALINE Learning "E"', ada_weights_e, target_e, learning_rate, tolerance, guess_test_ada_1_e, incorrect_test_ada_1_e,
  guess_test_ada_2_e, incorrect_test_ada_2_e, guess_test_ada_3_e, incorrect_test_ada_3_e) 

##################################################################################
# Use function from above to train Adaline for class A

 

if __name__ == "__main__":
	main()

###############################################################################


