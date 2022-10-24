from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


# TODO: add more algorithms + finish functions
def sgd_classifier(train_vectors, train_target):
    """sgd classifying algorithm

    :param train_vectors: list with the training feature data
    :param train_target: list with the training labels
    :return: clf: SGDClassifier, on training data fitted model
    """
    # mit Gradientenverfahren trainieren
    clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
    clf.fit(train_vectors, train_target)
    return clf


def svm_classifier(train_vectors, train_target):
    """svm classifying algorithm

        :param train_vectors: list with the training feature data
        :param train_target: list with the training labels
        :return: clf: SVC, on training data fitted model
        """
    clf = SVC(gamma='auto')
    clf.fit(train_vectors, train_target)
    return clf


def linear_regression(train_vectors, train_target):
    """linear regression

        :param train_vectors: list with the training feature data
        :param train_target: list with the training labels
        :return: reg: LinearRegression, on training data fitted model
        """
    reg = LinearRegression().fit(train_vectors, train_target)
    return reg


def perceptron(train_vectors, train_target):
    """perceptron algorithm

        :param train_vectors: list with the training feature data
        :param train_target: list with the training labels
        :return: clf: Perceptron, on training data fitted model
        """
    clf = Perceptron()
    clf.fit(train_vectors, train_target)
    return clf
