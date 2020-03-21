import random as rand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

# размер обучающей выборки
TRAIN_SET_SIZE = 10000

# размер тестовой выборки
TEST_SET_SIZE = 100


# функция, которую необходимо аппроксимировать
def approximation_function(x):
    return 4*x*x+5*x-3


# генерация рандомного набора значений x
def generate_x_array(length):
    return np.array([rand.uniform(0.5, 1) for _ in range(0, length)])


# генерация рандомной обучающей выборки
def generate_train_set(size):
    x_train = generate_x_array(size).reshape(-1, 1)
    y_train = np.array([approximation_function(el) for el in x_train]).reshape(-1, 1)

    return x_train, y_train


# генерация рандомной тестовойвыборки
def generate_test_set(size):
    x_test = generate_x_array(size).reshape(-1, 1)
    y_test = np.array([approximation_function(el) for el in x_test]).reshape(-1, 1)

    return x_test, y_test


# создание нейросети, ее обучение
def get_trained_neural_network(x_train, y_train):
    clf = MLPRegressor(hidden_layer_sizes=(20,), activation="logistic")
    clf.fit(x_train, y_train)
    return clf


# графическое отображение результатов
def show_results(x_test, y_test, y_predicted):
    fig, ax = plt.subplots()

    ax.plot(x_test, y_test, 'r.')
    ax.plot(x_test, y_predicted, 'b.')
    plt.legend(('real', 'predicted'), loc='upper right')

    ax.grid()
    plt.show()


# main
if __name__ == '__main__':
    x_train, y_train = generate_train_set(TRAIN_SET_SIZE)
    x_test, y_test = generate_test_set(TEST_SET_SIZE)

    neural_network = get_trained_neural_network(x_train, y_train)
    print("Predictions score: ", neural_network.score(x_test, y_test))

    y_predicted = neural_network.predict(x_test)
    show_results(x_test, y_test, y_predicted)
