from sklearn.model_selection import train_test_split
import vectorize_data
import training_algorithms
import joblib


class Training:
    """
    training class for a specific keyword
    """

    def __init__(self, data, topic):
        self.data = data
        self.topic = topic

    def train_for_one_keyword(self):
        """creates a train test split for the data, vectorizes and uses a training algorithm

        :return: classifier, SGDClassifier, trained model on the keyword
        """
        train, test, train_target, test_target = self.get_split()
        train_vec, test_vec, vectorizer, selector = vectorize_data.ngram_vectorize(train, train_target, test)
        joblib.dump(vectorizer, self.topic + "Vectorizer.pkl")
        joblib.dump(selector, self.topic + "Selector.pkl")
        classifier = training_algorithms.sgd_classifier(train_vec, train_target)

        return classifier

    def get_split(self):
        """creates a train test split

        :return: train: list, data to train
                 test: list, test data for evaluation
                 train_target: list, labels for training
                 test_target: list, test labels for evaluation
        """
        train, test, train_target, test_target = train_test_split(self.data["text"], self.data[self.topic],
                                                                  test_size=0.2, random_state=1)
        return train, test, train_target, test_target

    def get_x_val(self):
        """gets the transformed test list

        :return: list, with tfidf-transformed test list
        """
        return vectorize_data.ngram_vectorize(self.get_split()[0], self.get_split()[2], self.get_split()[1])[1]
