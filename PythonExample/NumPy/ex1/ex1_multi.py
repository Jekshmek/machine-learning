import numpy as np
# noinspection PyUnresolvedReferences
from featureNormalize import feature_normalize
# noinspection PyUnresolvedReferences
from gradientDescentMulti import gradient_descent_multi
# noinspection PyUnresolvedReferences
from normalEqn import normal_eqn

'''
    Linear regression with multiple variables
'''


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming
def ex1_multi():
    #data = np.loadtxt('ex1data2test.txt', delimiter=',')
    #X = np.array(data[:, :2])
    #X_normalize,mu, sigma = feature_normalize(X)
    #print(X_normalize)
    #return;
    # ----------------------------------------------
    # Using gradient descent
    # ----------------------------------------------
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    #data = np.loadtxt('test2.txt', delimiter=',')
    X = np.array(data[:, :2])
    y = np.array(data[:, 2])
    m = len(y)

    print('Первые 10 примеров из набора данных:')
    print( "X={}".format(X[0:10]))
    print( "y={}".format(y[0:10]))
    
    # feature normalization
    X_normalize,mu, sigma = feature_normalize(X)
    print('Первые 10 примеров из набора нормализованных данных:')
    print( X_normalize[0:10] )
    
    y = y.reshape(len(y), 1)
    # add intercept term to X
    X_normalize = np.column_stack((np.ones((m, 1)), X_normalize))

    # gradient descent
    alpha = 0.1
    num_iters = 400
    theta = np.zeros((3, 1))

    theta, j = gradient_descent_multi(X_normalize, y, theta, alpha, num_iters)
    print('theta =[', theta[0, 0], ',', theta[1, 0], ',', theta[2, 0], ']')
    #price = np.dot([1, 1650, 3], theta)[0]
    #features = np.array([17, 4], dtype=np.int32)
    features = np.array([1650, 3], dtype=np.int32)
    normal_features =  (features-mu)/sigma
    normal_features =  np.concatenate(( np.array([1]),normal_features))
    price=np.sum(normal_features* theta[:,0])
    print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {price}')

    # ----------------------------------------------
    # Using normal equation
    # ----------------------------------------------
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    #data = np.loadtxt('test2.txt', delimiter=',')
    X = np.array(data[:, :2])
    y = np.array(data[:, 2])
    m = len(y)

    # add intercept term to X
    X = np.column_stack((np.ones((m, 1)), X))
    y = y.reshape(len(y), 1)
    theta = normal_eqn(X, y)

    print('theta =[', theta[0, 0], ',', theta[1, 0], ',', theta[2, 0], ']')
    price = np.dot([1, 1650, 3], theta)[0]
    #price = np.dot([1, 17, 4], theta)[0]
    print(f'Predicted price of a 1650 sq-ft, 3 br house (using normal equation): {price}')
