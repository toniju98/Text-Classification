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

    def plot_learning_curve(self, title, X, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        self : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        axes : array of 3 axes, optional (default=None)
            Axes to use for plotting the curves.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt
