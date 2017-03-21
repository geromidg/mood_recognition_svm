#!/usr/bin/python
# -*- coding: utf8 -*-

from sys import argv

from os.path import join, splitext, basename
from glob import glob
from gc import collect as collect_garbage
from multiprocessing import cpu_count
from time import time

from extract_features import ExtractFeatures
from train_and_test import TrainAndTest

class Mirto:
    def __init__(self, dataset_paths, num_processes=None):
        self.dataset_paths = dataset_paths
        self.num_processes = num_processes

        if not self.num_processes:
            self.num_processes = cpu_count() - 1

    def run(self, all_classifiers=False):
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
    