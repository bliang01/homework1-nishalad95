"""Example unit tests for Homework 1

Important
=========

Do not modify the way in which your solution functions are called or imported.
The actual test suite used to grade your homework will import and call your
functions in the exact same way.

"""

# the unit test module
import unittest

# some useful functions from Numpy for creating your own tests
import numpy
from numpy import sin, cos, exp, pi, dot, eye, zeros, ones, array
from numpy.linalg import norm
from numpy.random import randn

# import the homework functions
from homework1.exercise1 import collatz_step, collatz
from homework1.exercise2 import gradient_step, gradient_descent
from homework1.exercise3 import (
    decompose,
    is_sdd,
    jacobi_step,
    jacobi_iteration,
    gauss_seidel_step,
    gauss_seidel_iteration,
)

class TestExercise1(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise1.collatz_step
    * homework1.exercise1.collatz

Here we list some suggested tests you should write for your code.
    """
    def test_collatz_step(self):
        # you should probably write some tests to determine if collatz_step is
        # being correctly computed
	self.assertEqual(collatz_step(1), 1)
	self.assertEqual(collatz_step(3), 10)
	self.assertEqual(collatz_step(2), 1)
	self.assertEqual(collatz_step(5), 16)
	self.assertEqual(collatz_step(4), 2)

    def test_collatz_step_one(self):
        # you should probably write a test to see if collatz_step handles the
        # n=1 case correctly
        self.assertEquals(collatz_step(1), 1)

    def test_collatz_step_error(self):
        # this test has been written for you. it demonstrates how to test if a
        # function raises an error
        with self.assertRaises(ValueError):
            collatz_step(-1)
            collatz_step(-2)
	    collatz_step(-2.5)
	with self.assertRaises(TypeError):
	    collatz_step(1.0)

    def test_collatz(self):
        # you should probably test the collatz() function against some collatz
        # sequences that you've computed by hand
	self.assertEquals(collatz(1), [1])
	self.assertEquals(collatz(2), [2, 1])
	self.assertEquals(collatz(3), [3, 10, 5, 16, 8, 4, 2, 1])
	with self.assertRaises(ValueError):
	    collatz(-1)
	    collatz(0)


class TestExercise2(unittest.TestCase):
    """Testing the validity of

    * hiomework1.exercise2.gradient_step
    * homework1.exercise2.gradient_descent

    In this problem we give less guidance on what tests you should write but
    the homework assignment document describes what we will be testing you on.
    It's up to you to write some good tests to make sure that your code works
    as intended.

    One simple test is, nonetheless, given for you.

    """
    def test_gradient_step(self):
        # this simple test determines if gradient_step is correctly computed on
        # a simple example: f(x) = x**2 - 1
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        x0 = 1
        x1 = gradient_step(x0, df, sigma=0.25)
        x1_actual = 0.5 # x0 - sigma*(2*x0)
        self.assertAlmostEqual(x1, x1_actual)

    def test2_sigmaAndEpsilon(self):
	# this test determines whether gradient_step and gradient descent raises
	# value errors on sigma and epsilon
	f = lambda x: x**2 - 1
        df = lambda x: 2*x
        x = 1
        with self.assertRaises(ValueError):
		gradient_step(x, df, sigma=1.5)
		gradient_step(x, df, sigma=-1)
		gradient_descent(f, df, x, sigma=0.5, epsilon=-1)
		gradient_descent(f, df, x, sigma=0.5, epsilon=2)
		gradient_descent(f, df, x, sigma=2, epsilon=0.1)
			
    
    def test3_convexfunctions(self):
        # this test verfies whether gradient_step works correctly for a variety of examples
        # verify the test on different function examples
        f = lambda x: x**4
        df = lambda x: 4*x**3
        x0 = 1
        x1 = gradient_step(x0, df, sigma=0.25)
        x1_actual = 0  # x0 - sigma*df(x0)
        self.assertAlmostEqual(x1, x1_actual)
 
        f = lambda x: exp(x)
        df = lambda x: exp(x)
        x0 = 1
        x1 = gradient_step(x0, df, sigma=0.9)
        x1_actual = -1.44  # x0 - sigma*df(x0)
        self.assertAlmostEqual(x1, x1_actual)

    def test4_simpleExamples(self):
	# this test verfies whether gradient_step works correctly for a variety of examples
	# verify the test on different function examples
	f = lambda x: x**3 - 5*x**2 + 4
	df = lambda x: 3*x**2 - 5*x
	x0 = 10
        x1 = gradient_step(x0, df, sigma=0.25)
        x1_actual = -52.5  # x0 - sigma*df(x0)
        self.assertAlmostEqual(x1, x1_actual)

	f = lambda x: sin(4*x)
        df = lambda x: 4*cos(x)
        x0 = 5
        x1 = gradient_step(x0, df, sigma=0.9)
        x1_actual = 3.979  # x0 - sigma*df(x0)
        self.assertAlmostEqual(x1, x1_actual) 

	f = lambda x: -x**2
        df = lambda x: -2*x
        x0 = 5
        x1 = gradient_step(x0, df, sigma=0.1)
        x1_actual = 6 # x0 - sigma*df(x0)
        self.assertAlmostEqual(x1, x1_actual) 

# Still need to implement many more tests!!!!!!


class TestExercise3(unittest.TestCase):
    """Testing the validity of

    * homework1.exercise3.decompose
    * homework1.exercise3.jacobi_step
    * homework1.exercise3.jacobi_iteration
    * homework1.exercise3.gauss_seidel_step
    * homework1.exercise3.gauss_seidel_iteration

    Some simple tests are already given for you but if you look closely they
    may not be sufficient to completely check the validity of your code since
    the example provided is too simple. A good test suite tests simple cases as
    well as more complex cases.

    """
    def test_decompose(self):
        # the test written below only tests if the identity matrix is properly
        # decomposed. this is not sufficient for testing if decompose() works
        # properly but is a good start.
        A = eye(3)
        D, L, U = decompose(A)
        D_actual = eye(3)
        L_actual = zeros((3,3))
        U_actual = zeros((3,3))
        self.assertAlmostEqual(norm(D_actual - D), 0)
        self.assertAlmostEqual(norm(L_actual - L), 0)
        self.assertAlmostEqual(norm(U_actual - U), 0)

        #test a 3 x 3 matrix
        A = array([[1,2,3], [4,5,6], [7,8,9]])
        D, L, U = decompose(A)
        D2 = numpy.diag([1,5,9])
        L2 = array([[0,0,0], [4,0,0], [7,8,0]]) 
        U2 = array([[0,2,3], [0,0,6], [0,0,0]])
        self.assertAlmostEqual(norm(D2 - D), 0)
        self.assertAlmostEqual(norm(L2 - L), 0)
        self.assertAlmostEqual(norm(U2 - U), 0) 

        #test a 4 x 4 matrix
        A = array([[1,2,3,4],  [5,6,7,8], [9,10,11,12], [13,14,15,16]])
        D, L, U = decompose(A)
        D4 = numpy.diag([1,6,11,16])
        L4 = array([[0,0,0,0], [5,0,0,0], [9,10,0,0], [13,14,15,0]])
        U4 = array([[0,2,3,4], [0,0,7,8], [0,0,0,12], [0,0,0,0]])
        self.assertAlmostEqual(norm(D4 - D), 0)
        self.assertAlmostEqual(norm(L4 - L), 0)
        self.assertAlmostEqual(norm(U4 - U), 0)

	#test a 1 x 1  matrix
        A = array([[10]])
        D, L, U = decompose(A)
        D5 = numpy.diag([10])
        L5 = None 
        U5 = None
        self.assertAlmostEquals(norm(D5 - D), 0)
        self.assertEquals(L5, None)
        self.assertEquals(U5, None)

    def test_jacobi_step(self):
        # the test written below only tests if jacobi step works in the case
        # when A is the identity matrix. In this case, jacobi_step() should
        # converge immediately the the answer. (Can you see why based on the
        # definition of Jacobi iteration?) This is not sufficient for testing
        # if jacobi_step() works properly but is a good start.
        D = eye(3)
        L = zeros((3,3))
        U = zeros((3,3))
        b = array([1,2,3])
        x0 = ones(3)
        x1 = jacobi_step(D, L, U, b, x0)

        self.assertAlmostEqual(norm(x1-b), 0)

    def test_jacobi_iteration(self):
        # the test written below only tests if jacobi iteration works in the
        # case when A is the identity matrix.
        A = eye(3)
        b = array([1,2,3])
        x0 = ones(3)
        x = jacobi_iteration(A, b, x0)

        self.assertAlmostEqual(norm(x-b), 0)



# The following code is run when this Python module / file is executed as a
# script. This happens when you enter
#
# $ python test_homework1.py
#
# in the terminal.
if __name__ == '__main__':
    unittest.main(verbosity=2) # run the above tests
