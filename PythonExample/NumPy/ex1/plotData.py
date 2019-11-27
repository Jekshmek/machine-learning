import matplotlib.pyplot as plt


# noinspection PyPep8Naming
def plot_data(X, y):
    plt.figure()
    plt.scatter(X, y, c='r', marker='x')
    plt.xlabel('Population of City')
    plt.ylabel('Profit in $')

    plt.show()
