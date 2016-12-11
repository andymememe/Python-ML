'''
[Get Dataset]
'''

import pandas as pd


def get_iris_data():
    '''
    Get iris Dataset

    Returns:
    iris : DataFrame

    Iris:
    -------------------------------------------------------------------------
    | Sepal length | Sepal width | Petal length | Petal width | Class label |
    |     float    |    float    |     float    |    float    |    string   |
    -------------------------------------------------------------------------
    * Class label : 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
    '''
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                       'machine-learning-databases/iris/iris.data',
                       header=None)
    return iris
