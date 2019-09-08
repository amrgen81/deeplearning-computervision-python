import numpy as np


# the benefit of yield is that we don't have to store the returned list to memory
# control return to the caller when encounter yield keyword
def generate(arr):
    for num in arr:
        # control return to the caller, then the caller can do their job on the return value
        yield num  # => control return to for loop body
        # at this point the caller finish one iteration on the element, control return
        print("generate::after yield {}".format(num))


array = [1, 5, 2, 7, 19, 3]
for num in generate(array):
    # => control gain from generator return by generate function, got num value
    print(num)
    print("main::after print {}".format(num))
    # => control go to the next yield
