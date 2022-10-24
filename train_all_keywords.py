from train_model import Training
from evaluate_model import EvaluationModel
import joblib


def train_all_topics(data, topics):
    """training for all topics and saving training model for each topic

    :param data: DataFrame, the dataset
    :param topics: list, all topics
    :return:
    """
    for topic in topics:
        training = Training(data, topic)
        classifier = training.train_for_one_keyword()
        joblib.dump(classifier, topic + "Model.pkl")


def evaluate_all_topics(data, topics):
    """evaluation for every topic

    :param data: DataFrame, the dataset
    :param topics: list, all topics
    :return:
    """
    for topic in topics:
        evaluation = EvaluationModel(data, topic)
        evaluation.evaluate_testdata()
