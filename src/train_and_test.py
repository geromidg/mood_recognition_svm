"""
.. module:: train_and_test
   :platform: Unix, Windows
   :synopsis: Contains methods to train and test on extracted features from an audio track.

.. moduleauthor: Dimitris Geromichalos <geromidg@gmail.com>
"""

from warnings import simplefilter

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

class TrainAndTest:
    """
    A class for training and testing on a dataset with extracted audio features.

    It holds the input dataset in the form of a feature matrix.
    The sklearn library is utilize for machine learning.
    The training is done using a pipeline that consists of the feature selector and the classifier.
    Hyperparameter optimization is done on the estimator with grid search on specific ranges.
    The tessting if done using k-fold cross validation.

    Args:
        feature_matrix (FeatureMatrix): The input dataset from which the featues will be extracted.
        sample_labels (List[str]): The labels of each track of the dataset.
        num_processes (Optional[int]): The number of processes to be spawned. Defaults to 8.
        folds (Optional[int]): The number of folds used in cross validation. Defaults to 10.

    Attributes:
        features (NumPy Array): The input dataset from which the featues will be extracted.
        labels (NumPy Array): The labels of each track of the dataset.
        num_processes (int): The number of processes to be spawned.
        folds (int): The number of folds used in cross validation.
        classifiers (Dict): The available classifiers (NB, KNN, SVM).
        knn_n_neighbors (List[int]): The number of nearest neighbors for KNN (hyperparameter).
        svm_gamma (List): The gamma's of SVM (hyperparameter).
        svm_C (List): The C's of SVM (hyperparameter).
        selector_transforms (List): The available selectors (ANOVA, PCA).
        anova_percentiles (List): The percentiles of ANOVA (hyperparameter).
        pca_n_components (List): The first components of PCA (hyperparameter).
    """

    simplefilter('ignore')

    def __init__(self, feature_matrix, sample_labels, num_processes=8, folds=10):
        self.features = feature_matrix.data
        self.labels = np.array(sample_labels)
        self.num_processes = num_processes
        self.folds = folds

        # Classifiers
        self.classifiers = {'nb': GaussianNB(), 'knn': KNeighborsClassifier(n_neighbors=1), 'svm': SVC(kernel='rbf', C=300)}
        
        self.knn_n_neighbors = [1, 3]

        self.svm_gamma = [1e-3, 1e-4]
        self.svm_C = [10**-1, 1, 10, 100, 1000]

        # Selectors
        self.selector_transforms = {'anova': SelectPercentile(f_classif), 'pca': PCA()}
        
        self.anova_percentiles = (np.arange(11) * 10)[1:]
        self.pca_n_components = np.floor(np.linspace(1, self.features.shape[1], 10)).astype(int)

    def run(self):
        """
        Runs the train/test task using SVM with ANOVA.

        The classifier and selector were chosen after yielding the best results after experiments.

        Returns:
            str: The report of the testing done.
        """
        
        return self.run_classifier_with_selector('svm', 'anova')

    def run_all(self):
        """
        Runs the train/test task using every combination of classifier/selector.

        Returns:
            str: The report of the testing done for each combination.
        """
        
        report_all = []

        report_all.append(self.run_classifier_with_selector('nb'))
        report_all.append(self.run_classifier_with_selector('nb', 'pca'))
        report_all.append(self.run_classifier_with_selector('nb', 'anova'))

        report_all.append(self.run_classifier_with_selector('knn'))
        report_all.append(self.run_classifier_with_selector('knn', 'pca'))
        report_all.append(self.run_classifier_with_selector('knn', 'anova'))

        report_all.append(self.run_classifier_with_selector('svm'))
        report_all.append(self.run_classifier_with_selector('svm', 'pca'))
        report_all.append(self.run_classifier_with_selector('svm', 'anova'))

        return report_all

    def run_classifier_with_selector(self, classifier_name, selector_name=None):
        """
        Runs the train/test task using the given classifier/selector.

        First, a grid seach is done on the parameters of the pipeline (selector+classifier).
        Then, the input training data are fit on the estimator.
        Finally, k-fold cross validation is done on the resulting estimator.

        Args:
            classifier_name (str): The classifier to be used.
            selector_name (str): The selector to be used. Defaults to None.

        Returns:
            str: The report of the testing done.
        """
        
        if selector_name in self.selector_transforms.keys():
            print 'Runnning %s with %s...' % (classifier_name.upper(), selector_name.upper())
        else:
            print 'Runnning %s...' % (classifier_name.upper())

        pipeline = self.classifiers[classifier_name]

        # Perform grid search to tune params
        if selector_name in self.selector_transforms.keys() or classifier_name != 'nb':
            params = dict()

            # Classifier params
            if classifier_name == 'knn':
                params['n_neighbors'] = self.knn_n_neighbors
            elif classifier_name == 'svm':
                params['C'] = self.svm_C
                params['gamma'] = self.svm_gamma

            # If a selector was selected
            if selector_name in self.selector_transforms.keys():
                # Add the classifier namespace to each param
                for param in params.keys():
                    params[classifier_name + '__' + param] = params.pop(param)

                # Selector params
                if selector_name == 'pca':
                    params['pca__n_components'] = self.pca_n_components
                elif selector_name == 'anova':
                    params['anova__percentile'] = self.anova_percentiles

                pipeline = Pipeline([(selector_name, self.selector_transforms[selector_name]), (classifier_name, pipeline)])

            estimator = GridSearchCV(pipeline, params, n_jobs=self.num_processes)
            estimator.fit(self.features, self.labels)

        else:  # NB without selector -> no parameter to do grid search on
            estimator = pipeline

        scores, predicted = self.run_cross_validation(estimator)

        return self.get_total_report(classifier_name, scores, predicted, estimator, selector_name)

    def run_cross_validation(self, estimator):
        """
        Runs k-fold cross validation on the given estimator.

        Args:
            estimator (Estimator): The estimator to run cross validation with.

        Returns:
            NumPy Array: The scores of the cross validation.
            NumPy Array: The predicted labels.
        """
        
        if hasattr(estimator, 'best_estimator_'):
            estimator = estimator.best_estimator_

        scores = cross_val_score(estimator, self.features, self.labels, cv=self.folds, n_jobs=self.num_processes)
        predicted = cross_val_predict(estimator, self.features, self.labels, cv=self.folds, n_jobs=self.num_processes)

        return scores, predicted

    @staticmethod
    def accuracy_report(classifier_name, scores, estimator, selector_name=None):
        """
        Calculates the accuracy of the estimator from the cross validation.

        Args:
            classifier_name (str): The classifier's name.
            scores (NumPy Array): The scores of the cross validation.
            estimator (Estimator): The estimator used in the process.
            selector_name (str): The selector's name. Defaults to None.

        Returns:
            str: The result of the cross validation.
        """
        
        report = '%s accuracy: %0.2f (+/- %0.2f)\n' % (classifier_name.upper(), scores.mean(), scores.std())

        if selector_name:
            if selector_name == 'pca':
                report += 'PCA components: %d' % (estimator.best_estimator_.named_steps['pca'].n_components)
            elif selector_name == 'anova':
                report += 'ANOVA percentile: %d' % (estimator.best_estimator_.named_steps['anova'].percentile)
            report += '\n'

        return report

    @staticmethod
    def pretty_cm(labels, predicted):
        """
        Calculates and prettifies a confusion matrix from the true/predicted labels.

        Args:
            labels (NumPy Array): The true labels.
            predicted (NumPy Array): The predicted labels.

        Returns:
            str: The prettified confusion matrix.
        """
        
        label_names = sorted([i for i in set(labels)])

        column_width = max([len(x) for x in label_names] + [5])
        
        cm = confusion_matrix(labels, predicted)
        new_cm = ''

        # Header
        new_cm += '    ' + ' ' * column_width
        for label in label_names:
            new_cm += '%{0}s'.format(column_width) % label
        new_cm += '\n'
        
        # Rows
        for i, label in enumerate(label_names):
            new_cm += '    %{0}s'.format(column_width) % label
            for j in range(len(label_names)):
                new_cm += '%{0}d'.format(column_width) % cm[i, j]
            new_cm += '\n'

        return new_cm

    def get_total_report(self, classifier_name, scores, predicted, estimator, selector_name, verbose=False):
        """
        Gathers all the info from the process and reports back.

        Args:
            classifier_name (str): The classifier's name.
            scores (NumPy Array): The scores of the cross validation.
            predicted (NumPy Array): The predicted labels.
            estimator (Estimator): The estimator used in the process.
            selector_name (str): The classifier's name.
            verbose (bool): Whether to print with detail. Defaults to False.


        Returns:
            str: The final report.
        """
        
        if verbose:
            total_report = '\n'
            total_report += self.accuracy_report(classifier_name, scores, estimator, selector_name)
            total_report += classification_report(self.labels, predicted)
            total_report += self.pretty_cm(self.labels, predicted)
        else:
            name = '%s+%s' % (classifier_name.upper(), selector_name.upper()) if selector_name else classifier_name.upper()
            cm = confusion_matrix(self.labels, predicted)
            label_names = sorted([i for i in set(self.labels)])
            total_report = (name, scores.mean() * 100, scores.std() * 100, (cm, label_names))

        return total_report
