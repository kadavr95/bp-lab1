import random as rand

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import axes3d, Axes3D

# размер обучающей выборки
TRAIN_SET_SIZE = 10000

# размер тестовой выборки
TEST_SET_SIZE = 500


# функция, которую необходимо аппроксимировать
def approximation_function(x1, x2):
    return -x1+5*x2-x1*x1-x2*x2+x1*x2


# генерация рандомного набора значений x1, x2
def generate_x_array(length):
    arr = []
    for i in range(0, length):
        arr.append([rand.uniform(0, 2), rand.uniform(0, 3)])

    return arr


# генерация рандомной обучающей выборки
def generate_train_set(size):
    x_train = generate_x_array(size)
    y_train = np.array([approximation_function(el[0], el[1]) for el in x_train])

    return x_train, y_train


# генерация рандомной тестовойвыборки
def generate_test_set(size):
    x_test = generate_x_array(size)
    y_test = np.array([approximation_function(el[0], el[1]) for el in x_test])

    return x_test, y_test


# создание нейросети, ее обучение
def get_trained_neural_network(x_train, y_train):
    clf = MLPRegressor(hidden_layer_sizes=(20,), activation="logistic")
    clf.fit(x_train, y_train)
    return clf


# графическое отображение результатов
def show_results(x_test, y_test, y_predicted):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([i[0] for i in x_test], [i[1] for i in x_test], y_test, c='r')
    ax.scatter([i[0] for i in x_test], [i[1] for i in x_test], y_predicted, c='b')

    plt.legend(('real', 'predicted'), loc='upper right')

    plt.show()


# main
if __name__ == '__main__':
    x_train, y_train = generate_train_set(TRAIN_SET_SIZE)
    x_test, y_test = generate_test_set(TEST_SET_SIZE)

    neural_network = get_trained_neural_network(x_train, y_train)
    print("Predictions score: ", neural_network.score(x_test, y_test))

    y_predicted = neural_network.predict(x_test)
    show_results(x_test, y_test, y_predicted)
