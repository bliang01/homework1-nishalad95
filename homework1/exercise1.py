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

    Returns
    -------
    int
        The result of C(n).

    """
    if n <= 0:
	raise ValueError('Value not accepted, please enter a positive value')
    elif n % 2 == 0:
	return n / 2
    elif n % 2 == 1 and n != 1:
	return 3*n + 1
    else:
	return 1

#print "n = 1: ", collatz_step(1)
#print "n = 3: ", collatz_step(3)
#print "n = 2: ", collatz_step(2)
#print "n = 5: ", collatz_step(5)
#print "n = 4: ", collatz_step(4)
#print "n = 1.0: ", collatz_step(1.0)
#print collatz_step(-1)


def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`. Calls
    `collatz_step` at each iteration.

    Parameters
    ----------
    n : int

    Returns
    -------
    sequence : list
        A Collatz sequence.

    """
    
    sequence = [n]
    currentValue = n    
    while (currentValue != 1):
	nextValue = collatz_step(currentValue)
	sequence.append(nextValue)
	currentValue = nextValue
    return sequence

#print "n = 1: ", collatz(1)
#print "n = 6: ", collatz(6)
#print "n = 2: ", collatz(2)
#print "n = 3: ", collatz(3)
#print "n = 0: ", collatz(0)
#print "n = -1: ", collatz(-1)
