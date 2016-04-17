# Author: Nisha LAD
# AMATH 483B: High Performance Scientific Computing
# Homework 1 Exercise 3
# Functions below are used to solve the linear systems of equations given an n x n matrix
# using the Jacobi and Gauss-Seidel Methods

# Hint: you should only need the following functions from numpy and scipy
from numpy import diag, tril, triu, dot, array, arange
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def decompose(A):
    """Returns three 2D matrices in an array.
    
    Parameters
    ----------
    A : ndarray
        2 dimensional n x n square matrix to be decomposed.
 
    Returns
    -------
    D, L, U : tuple
              Consisting off matrices D, U and L
    	      D : numpy.ndarray of ints or floats
              The 2 dimensional n x n square matrix consisting 
              of the leading diagonal elements of the matrix A
    	      L : numpy.ndarray of ints or floats
              The 2 dimensional n x n square matrix consisting 
              of the elements above the leading diagonal in matrix A
              U : numpy.ndarray of ints or floats
              The 2 dimensional n x n square matrix consisting 
              of the elements below the leading diagonal in matrix A
    """
    A = A.astype(float)
    D = diag(diag(A))
    L = tril(A, -1)
    U = A - L - D
    return D, L, U

A = array([[10, 2], [3, 11]])
D, L, U = decompose(A)
print D
print L
print U
print type(decompose(A))


def is_sdd(A):
    """Returns true if the matrix A is strictly diagonally dominant,
       returns false otherwise.

    A strictly diagonally dominant matrix is one which the absolute value of
    each element in the leading diagonal is greater than the sum of all other elements
    in the matrix excluding the leading diagonal terms.
    
    Parameters
    ----------
    A : numpy.ndarray of ints or floats
        2 dimensional n x n square matrix to be decomposed.
 
    Returns
    -------
    boolean: True if matrix A is strictly diagonally dominant, false otherwise
      
    """
    off_diag_elements = 0;
    for i in range(0, len(A), 1):
	for j in range (0, len(A), 1):
		if i != j:
			off_diag_elements += A[i,j] 

    for i in range(0, len(A), 1):
	if not abs(A[i, i]) > off_diag_elements:
		return False
    return True


def jacobi_step(D, L, U, b, xk):
    """Returns the next iteration xk1, given previous iteration xk, 
    using the jacobi iteration technique to solve linear equations numerically.
    
    Parameters
    ----------
    
 
    Returns
    -------
    xk1
      
    """
    A = D + L + U
    if is_sdd(A) == False:
	raise ValueError('Matrix A is not strictly diagonally dominant')

    # solve S(x(k+1)) = b - Tx(k)
    # S is diagonal matrix; inverse of S is the reciprocal of all leading diagonal elements
    T = L + U
    xk = xk.transpose()
    b = b.transpose()
    difference = b - dot(T, xk)
    for i in range(0, len(A), 1):
	D[i,i] = 1/D[i,i]
    S_inverse = D
    xk1 = dot(S_inverse, difference)
    xk1 = xk1.transpose()
    return xk1

xk = array([5, 6])
print jacobi_step(D, L, U, xk, xk)

def jacobi_iteration(A, b, x0, epsilon=1e-8):
    D, L, U = decompose(A)
    xk1 = jacobi_step(D, L, U, b, x0)
    while (norm(xk1 - x0, 2) < epsilon):
	x0 = xk1
	xk1 = jacobi_step(D, L, U, b, x0)
    return xk1

x0 = array([1, 2])
print jacobi_iteration(A, xk, x0)

def gauss_seidel_step(D, L, U, b, xk):
    pass

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    pass
