#!/usr/bin/python

"""
.. module:: main
   :platform: Unix, Windows
   :synopsis: The main module of the Mirto audio classification framework.

.. moduleauthor: Dimitris Geromichalos <geromidg@gmail.com>
"""

from sys import argv

from os.path import join, splitext, basename
from glob import glob
from gc import collect as collect_garbage
from multiprocessing import cpu_count
from time import time

from extract_features import ExtractFeatures
from train_and_test import TrainAndTest

class Mirto:
    """
    Implements the audio classification framework.

    It is responsible for integrating the feature extraction and training/testing modules
    in order to run a whole expiriment.

    Args:
        dataset_paths (List[str]): The paths of the files contained in the input dataset.
        num_processes (Optional[int]): The number of processes to be spawned. Defaults to `cpu_count() - 1`.

    Attributes:
        dataset_paths (List[str]): The paths of the files contained in the input dataset.
        num_processes (Optional[int]): The number of processes to be spawned.
    """

    def __init__(self, dataset_paths, num_processes=None):
        self.dataset_paths = dataset_paths
        self.num_processes = num_processes

        if not self.num_processes:
            self.num_processes = cpu_count() - 1

    def run(self, all_classifiers=False):
        """
        Runs the audio classification task.

        First, all the input audio tracks are converted from .au to .wav.
        Then, the features are extracted from the audio tracks in the form of a feature matrix.
        Finally, the feature matrix if fed to the train_and_test module for
        feature selection and training/testing.
		Garbage collection is done after each step.

        Args:
            all_classifiers (bool): Whether to run using all the available classifiers. Defaults to False.

        Returns:
            str: The final report of the train/test test.
            	It contains the classification's accuracy, among other info.
        """
        
        dataset = []
        for dataset_path in self.dataset_paths:
            dataset.append(glob(join(dataset_path, '*.wav')))
        dataset = sorted([item for sublist in dataset for item in sublist])

        labels = [splitext(basename(filename))[0].split('.')[0] for filename in dataset]

        t = time()
        extract_features = ExtractFeatures(dataset, num_processes=self.num_processes)
        feature_matrix = extract_features.run()
        print "Feature extraction took %f seconds" % (time() - t)

        collect_garbage()

        t = time()
        train_and_test = TrainAndTest(feature_matrix, labels, num_processes=self.num_processes)
        if all_classifiers:
            report = train_and_test.run_all()
        else:
            report = train_and_test.run()
        print "Training and testing took %f seconds" % (time() - t)

        collect_garbage()

        return report

if __name__ == "__main__":
    if len(argv) == 1:
        dataset_path = './dataset/training_set/'
    else:
        dataset_path = argv[1]

    print Mirto([dataset_path]).run(all_classifiers=True)
    