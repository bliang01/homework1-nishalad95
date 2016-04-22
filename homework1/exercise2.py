# Author: Nisha LAD
# AMATH 483B: High Performance Scientific Computing
# Homework 1 Exercise 2
# Functions create the gradient descent method of finding a local minimum


def gradient_step(xk, df, sigma):
    """Function calculates the next iteration of the gradient descent   
    
    Parameters
    ----------
    xk : double
	 Initial guess of local minimum
    df : lamda function
	 derivative of the function to minimize in terms of x; f'(x)
    sigma : double
	    scaling factor, an element of the set (0, 1)
	    if sigma is not within the set (0, 1) throws ValueError
       
    Returns
    -------
    xk - sigma*df(xk) : double 
			The next iteration x_(k+1) given the previous iterate xk
    """    

    if sigma > 1 or sigma < 0:
	raise ValueError('Illegal value of sigma entered, ensure 0 <= sigma <= 1')
    return xk - sigma*df(xk)


def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """Returns a minima of `f` using the Gradient Descent method.

    A local minima, x*, is such that `f(x*) <= f(x)` for all `x` near `x*`.
    This function returns a local minima which is accurate to within `epsilon`.

    `gradient_descent` raises a ValueError if 0 > sigma or sigma > 1

    Parameters
    ----------
    f :	      lambda function
              Function to determine its local minimum
    df :      lamda function
              derivative of the function to minimize in terms of x; f'(x)
    x :	      double
	      initial guess of x*
    sigma :   double
              scaling factor, an element of the set (0, 1)
	      if sigma is not within the set (0, 1) throws ValueError
    epsilon : double
	      Convergence criterion; where epsilon is an element of the set (0, 1)
	      raises ValueError if epsilon > 1 or epsilon < 0
       
    Returns
    -------
    x_k1 :     double 
              The next iteration
    """ 

    if epsilon < 0 or epsilon > 1:
	raise ValueError('Illegal value for epsilon, ensure 0 <= epsilon <= 1')
    x_k1 = x * 1.0
    x_k = (x + 1) * 1.0
    while (abs(x_k1 - x_k) > epsilon):
	if (f(x_k1) < -1000):
		raise ValueError('There is no local minimum found in this function')
	x_k = x_k1
        x_k1 = gradient_step(x_k, df, sigma)
  
    # now test if it the stationary point is actually a minimum
    if (df(x_k1 - 0.01) < 0 and df(x_k1 + 0.01) > 0):
    	return x_k1
    elif (df(x_k1 - 0.01) > 0):
	return gradient_descent(f, df, x_k1 - 0.01)
    else:
	return gradient_descent(f, df, x_k1 + 0.01)	

#f = lambda x : x**2
#df = lambda x : 2*x
#print gradient_descent(f, df, 0.5)
#print gradient_descent(f, df, 0)

#f = lambda x : 0.25*x**4 - 0.5*x**2
#df = lambda x : x**3 - x
#print gradient_descent(f, df, 0)

#f = lambda x : x**3
#df = lambda x : 3*x**2
#print gradient_descent(f, df, 0)

#f = lambda x : -x**2
#df = lambda x : -2*x
#print gradient_descent(f, df, 0)


