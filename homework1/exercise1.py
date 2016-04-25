# Author: Nisha LAD
# AMATH 483B: High Performance Scientific Computing
# Homework 1 Exercise 1
# Functions created to output a Collatz Sequence

def collatz_step(n):
    """Returns the result of the Collatz function.

    The Collatz function C : N -> N is used in `collatz` to generate collatz
    sequences. Raises an error if n < 1.

    Parameters
    ----------
    n : int
	to be passed through the collatz map.

    Returns
    -------
    C(n) : int
           The result of C(n), the first iteration of the collatz sequence of n.

    """
    if n <= 0:
	raise ValueError('Value not accepted, please enter a positive value')
    elif type(n) != int:
	raise TypeError('Type entered not accepted, please enter an integer')
    elif n % 2 == 0:
	return n / 2
    elif n == 1:
	return 1
    else:
	return 3*n + 1


def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`. Calls
    `collatz_step` at each iteration.

    Parameters
    ----------
    n : int
	to be passed through the collatz_step

    Returns
    -------
    sequence : list
               The Collatz sequence for starting point n

    """ 
    sequence = [n]
    currentValue = n
    # while the next value in sequence is not 1
    # retrieve next collatz map iteration
    # append to sequence array    
    while (currentValue != 1):
	nextValue = collatz_step(currentValue)
	sequence.append(nextValue)
	currentValue = nextValue
    return sequence


