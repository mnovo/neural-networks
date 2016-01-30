'''
Author: Michael Novo
email: mnovo006@fiu.edu
[EEL 5813] Neural Networks
Summer A 2015
Project 3 - Bidirectional Associative Memory (BAM!)
'''

from __future__ import division
import copy
import numpy as np
from math import exp


# Make arrays (vectors) for each of the "true" values for letters
# A, B, C, D, E, F, G, H patterns indices 0,1,2,3,4,5,6,7, respectively
patterns = np.array([[-1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1],   #A
		     [ 1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1,  1,  1, -1],   #B
		     [-1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1,  1,  1],   #C
 		     [ 1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1],   #D
 		     [ 1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1],   #E
		     [ 1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1],   #F
		     [-1,  1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1],   #G
		     [ 1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1],   #H
                     [ 1,  1, 0,  1, -1,  0,  1,  1, -1,  0, -1,  0,  1,  1, 0],   #Noisy B
                     [ 1,  0, -1,  0, -1,  1,  1, 0,  1,  1, -1,  1,  0,  1, -1],   #Noisy D
                     [ 1,  1,  0,  1, 0, -1,  0,  1, -1,  0, -1, -1,  0, -1, 0]])  #Noisy F

pattern_0 = [0,  0, 0,  0, 0,  0,  0,  0,  0,  0, 0,  0,  0, 0,  0]

# real_g = [-1,  1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1]

targets = np.array([	[-1, -1, -1],   #A
	                [-1, -1,  1],   #B
                        [-1,  1, -1],   #C
                        [-1,  1,  1],   #D
	                [ 1, -1, -1],   #E
                        [ 1, -1,  1],   #F
                        [ 1,  1, -1],   #G
	                [ 1,  1,  1],   #H
			[-1, -1,  1],   #B
			[-1,  1,  1],   #D
                        [ 1, -1,  1]])  #F

target_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'B', 'D', 'F']


# Adds the weight parameter provided and returns a new weight vector
def get_delta_w(x_input, single_target):
  delta_w = np.array([[0 for x in range(len(single_target))] for x in range(len(x_input))]) 
 
  for i in range(len(single_target)):
    delta_w[:,i] = np.dot(np.transpose(x_input), single_target[i]) 

  return delta_w

# Adds a weight vector and the delta_w vector
def add_weights(weights, delta_w):
  result = np.add(weights, delta_w)
  return result  

# Returns x*W
def present_x(input_x, weights):
  result = np.array([0 for x in range(len(weights[0]))])
  result = np.dot(input_x, weights)
  return result

# Returns y*W'
def present_y(input_y, weights):
  result = np.array([0 for x in range(len(weights))])
  result = np.dot(input_y, np.transpose(weights))
  return result

# Actication function for X and Y layers, seen on p142 Fausett
# Alters the input array by applying the activation function
def activation(input_array, theta):
  for i in range(len(input_array)):
    if (input_array[i] >  theta):
      input_array[i] = 1
    elif (input_array[i] <  theta):
      input_array[i] = -1
    else:
      input_array[i] = theta
  return input_array 

# Use previous activation to set yj by finding which indices have 0
# Once we know indices that have 0, then we use the previous activations
# to set the proper values. Return the activation result
def set_prev_acts(current, previous):
  for i in range(len(current)):
    if (current[i] == 0):
      current[i] = previous[i]
  return current

##################################################################################

def main():
  print 'Welcome to BAM!'
  # Initialize the proper size of the weight vector
  weights = np.array([[0 for x in range(len(targets[0]))] for x in range(len(patterns[0]))])
  #print weights
  theta = 0
  
  # Adds all weights, can also make this its own function
  # Step 0. 
  # Initialize the weights to store a set of P vectors
  for i in range(len(patterns)):
    #if (i in [1,3,5]):
      weights = add_weights(weights, get_delta_w(patterns[i], targets[i]))

  # Initialize all activations to 0
  # First need to present 0 x pattern, which serves as previous activation
  y_in_j = present_x(pattern_0, weights)
  y_j = activation(y_in_j, theta)
  y_j_prev = y_j

  x_in_i = present_y(y_j, weights)
  x_i = activation(x_in_i, theta)
  x_i_prev = x_i  
  
  # Step 1.
  # For each testing input, do Steps 2-6.
  for i in range(len(patterns)):    
    x_i_array = []    # Initialize an empty array to store previous activations of x
    y_j_array = []    # Initialize an empty array to store previous activations of y
    index = 0
    sentinel = 1  # If there is convergence, then sentinel change, and next pattern is tested
    index = 0     # Used within the while loop to keep track of how many iterations
    if (i > 100):   # Used to limit the number of patterns being tested
      sentinel = -1
    #if (i not in [1,3,5]):
    #  sentinel = -1
    else:
      print '' # At the beginning of each pattern presentation, print these stats
      print ''
      print '-------------------------------------------------------------------------------------------'
      print 'Pattern number: ' + str(i)
      print 'Pattern x: ' + str(patterns[i])    
      print 'Target y: ' + str(targets[i,:])
      print 'Target letter: ' + str(target_letters[i]) 

   # Step 2a.
   # Present input pattern x to the X-layer
    x_i_prev = patterns[i,:]
   # print 'Initial activation of X layer: ' + str(x_i_prev)

   # Step 2b.
   # Present input pattern y to the Y-layer
    y_j_prev = targets[i,:]
   # print 'Initial activation of Y layer: ' + str(y_j_prev)

   # Step 3.
   # While activations not converged, do Steps 4-6
    while sentinel == 1:
      index += 1      
      print ''
      print 'Iteration: ' + str(index)
      #print 'x_i_array: ' + str(x_i_array)
      #print 'y_j_array: ' + str(y_j_array)
      #####################################################################
      # Step 4. 
      # Update activations of units in the Y-layer.
      
      # Compute net inputs:
      if index == 1:
        y_in_j = present_x(patterns[i], weights) 
      else: 
        y_in_j = present_x(x_i_prev, weights) 

      # Compute activations:
      y_j = activation(y_in_j, theta)
      #print 'y_j after act:      ' + str(y_j)
      y_j = set_prev_acts(y_j, y_j_prev)
      print 'y_j (post activation): ' + str(y_j)

      # Send signal to X-layer.    
     #####################################################################
      # Step 5. 
      # Update activations of units in the X-layer.
      
      # Compute net inputs:
      x_in_i = present_y(y_j, weights)

      # Compute activations:
      x_i = activation(x_in_i, theta)
      #print 'x_i after act:      ' + str(x_i) 
      x_i = set_prev_acts(x_i, x_i_prev)
      print 'x_i (post activation): ' + str(x_i)

      # Send signal to Y-layer.
    

      ###################################################################
      # Step 6.
      # Test for convergence:
      # If the activations vectors x and y have reached equilibrium, then stop;
      # otherwise, continue
 
      for j in range(len(x_i_array)):
        if np.array_equal(x_i, x_i_array[j]):
          # If the x_activations are equal, make sure that the y_acts are also equal at same place
          if np.array_equal(y_j, y_j_array[j]):
            if (i == 8):
              print 'x_i original:          ' + str(patterns[1])
	      print 'x_i noisy:             ' + str(patterns[i])
            elif (i == 9):  
              print 'x_i original:          ' + str(patterns[3])
	      print 'x_i noisy:             ' + str(patterns[i])
            elif (i == 10):  
              print 'x_i original:          ' + str(patterns[5])
	      print 'x_i noisy:             ' + str(patterns[i])
            else:
              print 'x_i original:          ' + str(patterns[i])
	    print "Convergence at iteration: " + str(index)    
            sentinel = -1  

      # Store previous activations as an array. This may be redundant, but lets
      # leave it like this for now
      x_i_array.append(x_i)
      x_i_prev = x_i
      y_j_array.append(y_j)
      y_j_prev = y_j
  


if __name__ == "__main__":
	main()

##################################################################################

