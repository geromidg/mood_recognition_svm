#!/usr/bin/python

"""
.. module:: comparison
   :platform: Unix, Windows
   :synopsis: Contains methods to compare the different classifiers used in Mirto.

.. moduleauthor: Dimitris Geromichalos <geromidg@gmail.com>
"""

from sys import argv
from os import listdir, path
from itertools import product
from matplotlib import pyplot as plt
import numpy as np

from main import Mirto

def main(genres_dir):
    """
    Executes all 3 comparisons (label complexity, classifier comparison, confusion matrix)
    on a given dataset.

    Args:
        genres_dir (str): The directory containing the dataset.
    """
        
	genres = [path.join(genres_dir, genre) for genre in listdir(genres_dir)]

	run_label_complexity(genres)
	run_classifier_comparison(genres)
	run_confusion_matrix(genres)

def run_label_complexity(genres):
    """
    Classifies the given dataset for different number of labels to show the increasing complexity.

    Args:
        genres (List[str]): The filepath of the different genres of the dataset.
    """
        
	classifier = ''
	accuracies = []
	num_genres_list = range(2, len(genres) + 1)

	for num_genres in num_genres_list:
		dataset_paths = [genres[num_genre] for num_genre in xrange(num_genres)]

		classifier, accuracy = Mirto(dataset_paths).run(all_classifiers=False)[:2]
		accuracies.append(accuracy)

	classifier = classifier.replace('+', ' with ')

	figure = plt.figure(facecolor='w', edgecolor='k')

	axes = figure.add_subplot(1, 1, 1)
	axes.plot(num_genres_list, accuracies, 'y.', linewidth=5, markersize=45)
	axes.plot(num_genres_list, accuracies, 'k-', label=classifier, linewidth=5, markersize=45)
	axes.legend(title='Estimator', loc='best', fontsize='small', shadow=True, framealpha=0.5)
	axes.grid(True)
	axes.set_ylim(top=100)
	axes.set_xticks([int(num_genres) for num_genres in num_genres_list])
	axes.set_xticklabels(num_genres_list)
	axes.set_xlabel('Number of genres')
	axes.set_ylabel('Accuracy')
	axes.set_title('Label Complexity', {'weight': 'bold', 'size': 20})
	axes.margins(0.05) # 5% padding in all directions
	figure.tight_layout()
	figure.savefig('label_complexity' + '.png', bbox_inches='tight')
	
def run_classifier_comparison(genres):
    """
    Classifies the given dataset using different classifier to test their accuracy.

    Args:
        genres (List[str]): The filepath of the different genres of the dataset.
    """
        
	figure = plt.figure(facecolor='w', edgecolor='k')
	figure.suptitle('Classifier Comparison', fontsize=25, fontweight='bold')

	axes = figure.add_subplot(2, 1, 1)
	draw_classifier_accuracy_barplot(axes, genres, 2)
	axes = figure.add_subplot(2, 1, 2)
	draw_classifier_accuracy_barplot(axes, genres, len(genres))

	figure.tight_layout()
	figure.subplots_adjust(top=0.83)
	figure.savefig('classifier_comparison' + '.png', bbox_inches='tight')

def draw_classifier_accuracy_barplot(axes, genres, num_genres):
    """
    Draws barplots on each axis that contain each classifier's accuracy.

    Args:
    	axes (Axes): The axes where the barplots will be drawn.
        genres (List[str]): The filepath of the different genres of the dataset.
        num_genres (int): The number of genres.
    """
        
	dataset_paths = [genres[num_genre] for num_genre in xrange(num_genres)]
	all_reports = Mirto(dataset_paths).run(all_classifiers=True)

	classifiers = [report[0] for report in all_reports]
	accuracies = [report[1] for report in all_reports]
	stds = [report[2] for report in all_reports]

	norm_index = [i for i in xrange(len(classifiers)) if '+' not in classifiers[i]]
	norm_acc = [accuracies[i] for i in norm_index]
	norm_std = [stds[i] for i in norm_index]

	pca_index = [i for i in xrange(len(classifiers)) if 'PCA' in classifiers[i]]
	pca_acc = [accuracies[i] for i in pca_index]
	pca_std = [stds[i] for i in pca_index]

	anova_index = [i for i in xrange(len(classifiers)) if 'ANOVA' in classifiers[i]]
	anova_acc = [accuracies[i] for i in anova_index]
	anova_std = [stds[i] for i in anova_index]

	ind = np.arange(3)
	width = 0.25

	bars_0 = axes.bar(ind + width * 0, norm_acc, width, color='y', yerr=norm_std)
	bars_1 = axes.bar(ind + width * 1, pca_acc, width, color='c', yerr=pca_std)
	bars_2 = axes.bar(ind + width * 2, anova_acc, width, color='m', yerr=anova_std)

	axes.set_xticks(ind + width * 1.5)
	axes.set_xticklabels(('Naive Bayes', 'K-Nearest Neighbors', 'SVM'))
	axes.set_ylim([0, 100])
	axes.set_xlabel('Classifiers', {'size' :15})
	axes.set_ylabel('Accuracies', {'size': 15})
	axes.set_title('Number of genres: %d' % (num_genres), {'size': 20})
	axes.title.set_position([.5, 1.06])
	axes.legend((bars_0[0], bars_1[0], bars_2[0]), ('None', 'PCA', 'ANOVA'), title='Selector', loc='best', fontsize='small', shadow=True, framealpha=0.5)

	for bars in [bars_0, bars_1, bars_2]:
	    for rect in bars:
	        height = rect.get_height()
	        axes.text(rect.get_x() + rect.get_width() / 2., 0.75 * height, '%d' % int(height), ha='center', va='bottom')

def run_confusion_matrix(genres):
    """
    Calculates the confusion matrix of the classification of the given dataset.

    Args:
        genres (List[str]): The filepath of the different genres of the dataset.
    """
        
	dataset_paths = [genres[num_genre] for num_genre in xrange(len(genres))]
	cnf_matrix, label_names = Mirto(dataset_paths).run(all_classifiers=False)[3]

	plot_confusion_matrix(cnf_matrix, label_names, title='Confusion Matrix')
	plot_confusion_matrix(cnf_matrix, label_names, title='Confusion Matrix (normalized)', normalize=True)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    Plots the given confusion matrix.

    Args:
        cm (str): The confusion matrix to be plotted.
		classes (List[str]): The classes (labels) of the confusion matrix.
		normalize (Optional[bool]): Whether to normalize the confusion matrix. Defaults to False.
		title (Optional[str]): The title of the confusion matrix. Defaults to 'Confusion matrix'.
    """
        
	cmap = plt.cm.Reds if normalize else plt.cm.Blues

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.colorbar()
	plt.ylabel('True Genre')
	plt.xlabel('Predicted Genre')
	plt.xticks(np.arange(len(classes)), classes, rotation=45)
	plt.yticks(np.arange(len(classes)), classes)
	plt.title(title)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > (cm.max() / 2.) else "black")

	plt.tight_layout()
	plt.savefig('confusion_matrix' + ('_normalized' * normalize) + '.png', bbox_inches='tight')

if __name__ == '__main__':
	if len(argv) != 2:
		print 'Usage: %s /path/to/dataset_directory/' % (argv[0])
		exit(-1)

	main(argv[1])
