"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file utilities.py
PURPOSE OF THE FILE: Provides utilities used from different files.py.
"""

import pickle
import random
import time
from statistics import mean, stdev

import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils.constants import table_performances, repetitions, table_performances_ensemble_learning, folds


# Main utilities

def read_csv_from(path_csv, separator=','):
    """Reads csv

    Args:
        path_csv (string): The path of csv
        separator (string): The separator character

    Return:
        Array[]: array
    """
    return np.array(pandas.read_csv(path_csv, sep=separator))


def write_csv_in(path_csv, dictionary):
    """Writes csv

    Args:
        path_csv (string): The path of csv
        dictionary (dict): The dictionary
    """
    df = pd.DataFrame(dictionary)
    df.to_csv(path_csv, index=False)


def serialize_object_in(path_object, obj):
    """Serialize object

    Args:
        path_object (string): The path of serialized object
        obj (object): The object
    """
    with open(path_object, "wb") as outfile:
        pickle.dump(obj, outfile)


def deserialize_object_from(path_object):
    """Deserialize object

    Args:
        path_object (string): The path of serialized object

    Return:
        object: The object
    """
    with open(path_object, "rb") as infile:
        object = pickle.load(infile)
    return object


def write_string_in(path_txt, string):
    """Writes txt

    Args:
        path_txt (string): The path of txt
        string (string): The string
    """
    file = open(path_txt, "w")
    file.write(string)
    file.close()


# Other utilities

def get_path_baseline_selected_features(config):
    """Returns path where csv of baseline's selected features are contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: The path of csv of baseline's selected features
    """
    return config['features']['dataset']['selected_features']


def get_path_tasks_selected_features(config):
    """Returns path where csv of tasks' selected features are contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: The path of csv of tasks' selected features
    """
    return config['features']['dataset_tasks']['selected_features']


def get_model_from(features_and_grid_search, name_model, random_state, feature_selection_technique=None):
    """Returns model with specific hyperparameters

    Args:
        features_and_grid_search (dict): Models' hyperparameters
        name_model (string): The name of the learning algorithm
        random_state (int): The initial condition of each model (where is possible)
        feature_selection_technique (string): The name of feature selection technique

    Return:
        Class: The model set with specific hyperparameters
    """
    if feature_selection_technique is not None:
        if feature_selection_technique.__contains__("RFE"):
            feature_selection_technique = "RFE"

    if name_model == "K-NN":
        if feature_selection_technique is not None:
            return KNeighborsClassifier(
                n_neighbors=features_and_grid_search["K-NN"][feature_selection_technique]["hyperparameters"][
                    "n_neighbors"])
        else:
            return KNeighborsClassifier(
                n_neighbors=features_and_grid_search["K-NN"]["hyperparameters"]["n_neighbors"])

    elif name_model == "SVM":
        if feature_selection_technique is not None:
            return SVC(
                kernel=features_and_grid_search["SVM"][feature_selection_technique]["hyperparameters"]["kernel"],
                C=features_and_grid_search["SVM"][feature_selection_technique]["hyperparameters"]["C"],
                gamma=features_and_grid_search["SVM"][feature_selection_technique]["hyperparameters"]["gamma"],
                random_state=random_state)
        else:
            return SVC(
                kernel=features_and_grid_search["SVM"]["hyperparameters"]["kernel"],
                C=features_and_grid_search["SVM"]["hyperparameters"]["C"],
                gamma=features_and_grid_search["SVM"]["hyperparameters"]["gamma"],
                random_state=random_state)

    elif name_model == "LR":
        if feature_selection_technique is not None:
            return LogisticRegression(
                C=features_and_grid_search["LR"][feature_selection_technique]["hyperparameters"]["C"],
                random_state=random_state)
        else:
            return LogisticRegression(
                C=features_and_grid_search["LR"]["hyperparameters"]["C"],
                random_state=random_state)

    elif name_model == "RF":
        if feature_selection_technique is not None:
            return RandomForestClassifier(
                n_estimators=features_and_grid_search["RF"][feature_selection_technique]["hyperparameters"][
                    "n_estimators"],
                max_depth=features_and_grid_search["RF"][feature_selection_technique]["hyperparameters"]["max_depth"],
                min_samples_split=features_and_grid_search["RF"][feature_selection_technique]["hyperparameters"][
                    "min_samples_split"],
                min_samples_leaf=features_and_grid_search["RF"][feature_selection_technique]["hyperparameters"][
                    "min_samples_leaf"],
                random_state=random_state)
        else:
            return RandomForestClassifier(
                n_estimators=features_and_grid_search["RF"]["hyperparameters"]["n_estimators"],
                max_depth=features_and_grid_search["RF"]["hyperparameters"]["max_depth"],
                min_samples_split=features_and_grid_search["RF"]["hyperparameters"]["min_samples_split"],
                min_samples_leaf=features_and_grid_search["RF"]["hyperparameters"]["min_samples_leaf"],
                random_state=random_state)


def update_table_experiment_from(table_experiments, performances):
    """Performs min accuracy, max accuracy, mean accuracy, accuracies' standard deviation and mean time

    Args:
        table_experiments (dict): The table of performances
        performances (List[]): Accuracies and Times produced by the experiment

    Return:
        dict: The table of performances
    """

    def extract_results_from(performances, decimal_digit=3):
        """Performs rounding of performances

        Args:
            performances (List[]): Accuracies and Times produced by the experiment
            decimal_digit (int): The decimal digit in order to round performances

        Return:
            float: Rounded min accuracy
            float: Rounded max accuracy
            float: Rounded mean accuracy
            float: Rounded standard deviation
            float: Rounded elapsed time
        """
        accuracies = performances[0:-1]
        mean_elapsed_time = performances[-1]
        min_accuracies = round(min(accuracies) * 100, decimal_digit)
        max_accuracies = round(max(accuracies) * 100, decimal_digit)
        mean_accuracies = round(mean(accuracies) * 100, decimal_digit)
        sdv_accuracies = round(stdev(accuracies) * 100, decimal_digit)
        mean_elapsed_time = round(mean_elapsed_time, 3)
        return min_accuracies, max_accuracies, mean_accuracies, sdv_accuracies, mean_elapsed_time

    min_accuracies, max_accuracies, mean_accuracies, sdv_accuracies, elapsed_time = extract_results_from(performances)
    table_experiments["min_accuracies[%]"].append(min_accuracies)
    table_experiments["max_accuracies[%]"].append(max_accuracies)
    table_experiments["mean_accuracies[%]"].append(mean_accuracies)
    table_experiments["sdv_accuracies[%]"].append(sdv_accuracies)
    table_experiments["time[s]"].append(elapsed_time)
    return table_experiments


def get_performances_by_k_fold_stratified_cross_validation_from(samples, labels, model):
    """Returns accuracies and time performed by K Fold Stratified Cross-Validation

    Args:
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        model (Class): The learning algorithm

    Return:
        List[]: Accuracies and time performed by K Fold Stratified Cross-Validation
    """

    def k_fold_stratified_cross_validation(samples, labels, model, n_splits=3, shuffle=True):
        """Performs K Fold Stratified Cross-Validation

        Args:
            samples (List[]): Features for each sample
            labels (List[]): The label for each sample
            model (Class): The learning algorithm
            n_splits (int): The number of folds
            shuffle (Bool): The flag of suffle before the splitting in folds

        Return:
            List[]: Accuracies performed by K Fold Stratified Cross-Validation
        """
        accuracies = []
        for index_train, index_test in StratifiedKFold(n_splits=n_splits, shuffle=shuffle).split(samples, labels):
            samples_train_fold, samples_test_fold = samples[index_train], samples[index_test]
            labels_train_fold, labels_test_fold = labels[index_train], labels[index_test]
            model.fit(samples_train_fold, labels_train_fold)
            accuracies.append(model.score(samples_test_fold, labels_test_fold))
        return accuracies

    start = time.time()
    partial_performances = k_fold_stratified_cross_validation(samples, labels, model)
    end = time.time()
    partial_performances.append(end - start)
    return partial_performances


# Baseline and Tasks Experiments

def clean_table_performances():
    """Clean table of performances

    Return:
        dict: Table of performances
    """
    table_performances["min_accuracies[%]"] = []
    table_performances["max_accuracies[%]"] = []
    table_performances["mean_accuracies[%]"] = []
    table_performances["sdv_accuracies[%]"] = []
    table_performances["time[s]"] = []
    return table_performances


def perform_experiment_all_features_from(all_features_configurations, samples, labels, name_model, random_state,
                                         table_experiments):
    """Performs experiment using all features

    Args:
        all_features_configurations (dict): Models' hyperparameters
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        name_model (string): The name of the learning algorithm
        random_state (int): The initial condition of each model (where is possible)
        table_experiments (dict): The table of performances

    Return:
        List[]: The table of performances
    """
    # Extracts configured models
    model = get_model_from(all_features_configurations, name_model, random_state)

    # Collects results for each repetition
    results = []
    sum_results = []
    for index_repetition in range(repetitions):

        # Performs K Fold Stratified Cross-Validation
        partial_results = get_performances_by_k_fold_stratified_cross_validation_from(samples, labels, model)
        index = 0
        for partial_result in partial_results:
            if index_repetition == 0:
                sum_results.append(partial_result)
            else:
                sum_results[index] += partial_result
                index += 1

    # Means results leis on repetitions
    for result in sum_results:
        results.append(result / repetitions)

    # Refines results into apposite dictionary
    return update_table_experiment_from(table_experiments, results)


def perform_experiment_selected_features_from(feature_selection_configurations, datasets, labels, name_model,
                                              random_state, table_experiments):
    """Performs experiment using selected features

    Args:
        feature_selection_configurations (dict): Models' hyperparameters
        datasets (List[]): Features selected for each sample for each technique
        labels (List[]): The label for each sample
        name_model (string): The name of the learning algorithm
        random_state (int): The initial condition of each model (where is possible)
        table_experiments (dict): The table of performances

    Return:
        dict: The table of performances
    """

    def get_dataset_from(datasets, name_model, feature_selection_technique):
        """Returns samples' selected features

        Args:
            datasets (List[]): Features selected for each sample for each technique
            name_model (string): The name of the learning algorithm
            feature_selection_technique (string): The name of feature selection technique

        Return:
            List[]: Samples' selected features
        """
        if feature_selection_technique == "CFS":
            return datasets[0][1]
        elif feature_selection_technique == "ReliefF":
            return datasets[1][1]
        elif feature_selection_technique == "RFE":
            if name_model == "SVM":
                return datasets[2][1]
            elif name_model == "LR":
                return datasets[3][1]
            elif name_model == "RF":
                return datasets[4][1]
        elif feature_selection_technique == "GRF":
            return datasets[5][1]

    # For each Feature Selection Technique
    feature_selection_techniques = ["CFS", "ReliefF", "RFE", "GRF"]
    for feature_selection_technique in feature_selection_techniques:
        results = []
        sum_results = []

        # Collects results for each repetition
        if not (name_model == "K-NN" and feature_selection_technique == "RFE"):
            for index_repetition in range(repetitions):
                # Extracts samples' selected features
                samples = get_dataset_from(datasets, name_model, feature_selection_technique)

                # Extracts configured models
                model = get_model_from(feature_selection_configurations, name_model, random_state,
                                       feature_selection_technique)

                # Performs K Fold Stratified Cross-Validation
                partial_results = get_performances_by_k_fold_stratified_cross_validation_from(samples, labels, model)
                index = 0
                for partial_result in partial_results:
                    if index_repetition == 0:
                        sum_results.append(partial_result)
                    else:
                        sum_results[index] += partial_result
                        index += 1

            # Means results leis on repetitions
            for result in sum_results:
                results.append(result / repetitions)
        else:
            results = [0.000, 0.000, 0.000, 0.000, 0.000]

        # Refines results into apposite dictionary
        table_experiments = update_table_experiment_from(table_experiments, results)
    return table_experiments


# Ensemble Learning Experiment
def clean_table_performances_ensemble_learning():
    """Clean table of performances for Ensemble Learning

    Return:
        dict: The table of performances for Ensemble Learning
    """
    table_performances_ensemble_learning["min_accuracies[%]"] = []
    table_performances_ensemble_learning["max_accuracies[%]"] = []
    table_performances_ensemble_learning["mean_accuracies[%]"] = []
    table_performances_ensemble_learning["sdv_accuracies[%]"] = []
    table_performances_ensemble_learning["time[s]"] = []
    return table_performances_ensemble_learning


def get_balanced_folds_from(samples, labels):
    """Generates balanced folds for K Fold Stratified Cross-Validation algorithm

    Args:
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample

    Return:
        Array[]: Folded samples
        Array[]: Folded labels
    """

    # Sets the index positions of the sample sparing the classes to the side
    positions = labels.argsort()
    samples = samples[positions[::-1]]
    labels = labels[positions[::-1]]

    # Generates balanced folds
    number_elements_per_fold = int(len(labels) / folds)
    folds_samples = []
    folds_labels = []
    index_forward = 0
    index_backward = -1
    for index_fold in range(0, folds):
        fold_samples = []
        fold_labels = []

        # Faces eventually up with the remaining samples of the division in folds
        if index_fold == folds:
            number_elements_per_fold += len(labels) % number_elements_per_fold

        # Partitions samples in the specific fold
        for num_element in range(0, number_elements_per_fold):
            if not index_fold % 2 == 0:
                num_element += 1
            if num_element % 2 == 0:
                fold_samples.append(samples[index_forward, :])
                fold_labels.append(labels[index_forward])
                index_forward += 1
            else:
                fold_samples.append((samples[index_backward, :]))
                fold_labels.append(labels[index_backward])
                index_backward -= 1

        # Fills balanced folds
        folds_samples.append(fold_samples)
        folds_labels.append(fold_labels)
    return np.array(folds_samples), np.array(folds_labels)


def get_random_dataset_samples_positions(num_samples=75):
    """Generates randomly the dataset's samples positions in order to produce a suffle

    Args:
        num_samples (int): The number of dataset's samples

    Return:
        List[]: Positions to suffle the dataset's samples
    """
    positions = []
    while len(positions) < num_samples:
        pos = random.randint(0, num_samples - 1)
        if pos not in positions:
            positions.append(pos)
    return positions


def get_list_accuracy_from(list_accuracy, results, labels_test_folds, vote_to_overcame=5):
    """Performs definitive prediction from Ensemble Learning composed by specific experts

    Args:
        list_accuracy (List[]): The accuracy where save the accuracy of the prediction
        results (List[]): Experts' votes
        labels_test_folds (List[]): The test's labels
        vote_to_overcame (int): The majority vote to overcame for confirm the prediction

    Return:
        List[]: Accuracies achieved by specific experts
    """
    majority_voting_results = [1 if vote >= vote_to_overcame else 0 for vote in results]
    list_accuracy.append(accuracy_score(labels_test_folds, majority_voting_results))
    return list_accuracy
