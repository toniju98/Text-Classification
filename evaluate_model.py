from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from train_model import Training
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


class EvaluationModel:
    """class for evaluating training model

    """

    def __init__(self, data, topic):
        self.data = data
        self.topic = topic
        self.training = Training(data, topic)

    def evaluate_testdata(self):
        """Evaluates the test data with different methods
        """
        classifier = self.training.train_for_one_keyword()
        predicted = classifier.predict(self.training.get_x_val())
        test_target = self.training.get_split()[3]
        print(metrics.accuracy_score(test_target, predicted))
        print(metrics.confusion_matrix(test_target, predicted))
        print(metrics.classification_report(test_target, predicted))
        print(cross_validate(self.get_pipeline(), self.data['text'],
                             self.data[self.topic],
                             scoring=['precision_macro', 'recall_macro'],
                             cv=5, return_train_score=False))

    # TODO: right implementation
    def grid_search(self):
        """Grid Search for finding the best parameters for the training

        :return:
        """
        # Parameter für die Grid-Suche festlegen...
        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5),
                      }
        # ...und suchen (kann dauern)
        gs_clf = GridSearchCV(self.get_pipeline, parameters, n_jobs=-1, scoring='accuracy').fit(
            self.training.get_split()[0],
            self.training.get_split()[2])
        # Ergebnisse für Testmenge vorhersagen
        predicted = gs_clf.predict(self.training.get_split()[1])
        # und deren Metriken berechnen
        print(metrics.classification_report(self.get_split()[3], predicted))
        # Gefundene Parameter anzeigen
        gs_clf.best_params_

    def get_pipeline(self):
        """creates a pipeline for better evaluation

        :return: text_pl, Pipeline
        """
        # Pipeline konstruieren, macht BOW, TF/IDF und Training
        text_pl = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, tol=1e-3))])
        text_pl.fit(self.training.get_split()[0], self.training.get_split()[2])
        return text_pl

