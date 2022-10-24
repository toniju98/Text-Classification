import joblib
import json


class Prediction:
    """class for predicting topic for dataset

    """
    def __init__(self, data, topics):
        self.data = data
        self.topics = topics

    def predict_data(self, topic, data2):
        """predicts if an article is about this specific topic or not

        :param topic: topic, which will be checked
        :param data2: dataset, that needs to be checked
        :return:
        """
        # TODO: fix problem with vectorizer
        # training = Training(self.data, topic)
        vectorizer = joblib.load(topic + "Vectorizer.pkl")
        tfidf_transform = vectorizer.transform(data2["text"])
        selector = joblib.load(topic + "Selector.pkl")
        tfidf_transform = selector.transform(tfidf_transform)
        clf = joblib.load(topic + "Model.pkl")
        predicted = clf.predict(tfidf_transform)
        data2[topic] = predicted

    def predict_whole_data(self, data):
        """predict_data for each topic

        :param data:
        :return:
        """
        for k in self.topics:
            self.predict_data(k, data)
            json_data = self.dataframe_to_json(self.filter_datasets(self.sort_data(k)))
            with open(k + '.json', 'w', encoding='utf-8') as outfile:
                json.dump(json_data, outfile, ensure_ascii=False, indent=4)

    def sort_data(self, topic):
        """sorting data for a specific topic

        :param topic:
        :return:
        """
        match_topic = (self.data[topic] == 1)
        data2 = self.data[match_topic]
        return data2

    def filter_datasets(self, data2):
        """

        :param data2:
        :return:
        """
        for k in self.topics:
            data2 = data2.drop(columns=[k])
        return data2

    def dataframe_to_json(self, data2):
        """

        :param data2:
        :return:
        """
        return data2.to_json(orient='records')

    def load_topic_list(self, path):
        """

        :param path:
        :return:
        """
        with open(path, encoding="utf8") as data_file:
            d = json.load(data_file)
        topic_list = d
        return topic_list
