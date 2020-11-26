# This is a sample Python script.
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

x = [1, 2, 3]
y = [-1, -2, -3]
print(x)
prediction1 = np.zeros(len(x))
print(prediction1)

for i, j in zip(x, y):
    print(i, j)
    # prediction1 += ((x)) / 5
    # print(prediction1)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
