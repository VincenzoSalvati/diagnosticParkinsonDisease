"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file tasks_final_ensemble_learning_experiment.py
PURPOSE OF THE FILE: Performs Ensemble Learning experiments.
"""

import os
import time

import numpy as np
from tqdm import tqdm

from experiments.tasks_final_ensemble_learning_experiment_best_hyperparameters import get_model_from
from utils.constants import ensemble_learning_configurations, repetitions, folds, classification_tasks
from utils.feature_selection import feature_selection
from utils.models_and_hyperparameters import get_models_selected_features
from utils.models_and_hyperparameters import get_samples_selected_features_from
from utils.utilities import clean_table_performances_ensemble_learning, write_csv_in, \
    get_path_tasks_selected_features, get_random_dataset_samples_positions, get_balanced_folds_from, \
    get_list_accuracy_from, update_table_experiment_from


def get_path_performances_ensemble_learning_default_hyperparameters(config):
    """Returns path where (ensemble_learning) performances of models using default hyperparameters has to be saved

    Args:
        config (dict): The instance of config.yml

    Return:
        string: The path where (ensemble_learning) performances of models using default hyperparameters has to be saved
    """
    return config['ensemble_learning_default_hyperparameters']


def tasks_final_ensemble_learning_experiment(config, dataset_tasks, random_state):
    """Performs models' performances of Ensemble Learning using default hyperparameters and save them into a csv

    Args:
        config (dict): The instance of config.yml
        dataset_tasks (List[][]): Features and labels for each sample for each task
        random_state (int): The initial condition of each model (where is possible)
    """
    # Trains Best for Task (BFT) for each composition of experts (from the best one until the worst one)
    performances_one_task, performances_two_tasks, performances_three_tasks, performances_four_tasks, \
        performances_five_tasks, performances_six_tasks, performances_seven_tasks, \
        performances_eight_tasks = [], [], [], [], [], [], [], []
    times_one_task, times_two_tasks, times_three_tasks, times_four_tasks, times_five_tasks, times_six_tasks, \
        times_seven_tasks, times_eight_tasks = 0, 0, 0, 0, 0, 0, 0, 0
    for _ in tqdm(range(repetitions), "Performing repetitions to be mediated (Ensemble Learning Experiment)"):
        results = []

        # Randoms the index positions of the sample for each repetition
        positions = get_random_dataset_samples_positions()

        # For each expert (from the best one until the worst one)
        for num_task in classification_tasks:
            samples, labels = np.array(dataset_tasks[num_task][0]), np.array(dataset_tasks[num_task][1])

            # Suffle of the sample
            samples = samples[positions[::-1]]
            labels = labels[positions[::-1]]

            # Fetches selected features for the specific task and models' initial conditions
            if not os.path.exists(get_path_tasks_selected_features(config) + str(num_task + 1) + "_random_state_" + str(
                    random_state)):
                print("Selected features on task " + str(num_task + 1) + " not detected." +
                      "\nPerforming Feature Selection Techniques...")
                feature_selection(config, samples, labels, random_state, num_task + 1)

            # Fetches the best configuration Model/Feature-Selection-Technique for the specific task
            _, feature_selection_configurations = get_models_selected_features(config, samples, random_state,
                                                                               num_task=num_task + 1)
            feature_selection_technique = ensemble_learning_configurations["task_" + str(num_task + 1)][
                "feature_selection_technique"]
            name_model = ensemble_learning_configurations["task_" + str(num_task + 1)]["model"]
            model = get_model_from(feature_selection_configurations, name_model, random_state,
                                   feature_selection_technique)

            # Performs K Fold Stratified Cross-Validation of BFT
            folds_samples, folds_labels = get_balanced_folds_from(samples, labels)
            labels_test_folds = []
            start = time.time()
            for index_test in range(0, folds):
                samples_train_fold, labels_train_fold = None, None
                for index in range(0, folds):
                    if not index == index_test:
                        if samples_train_fold is None:
                            samples_train_fold, labels_train_fold = folds_samples[index], folds_labels[index]
                        else:
                            samples_train_fold = np.concatenate((samples_train_fold, folds_samples[index]), axis=0)
                            labels_train_fold = np.concatenate((labels_train_fold, folds_labels[index]), axis=0)
                samples_test_fold, labels_test_fold = folds_samples[index_test], folds_labels[index_test]
                samples_train_fold = get_samples_selected_features_from(config, samples_train_fold, random_state,
                                                                        num_task + 1, feature_selection_technique)
                model = model.fit(np.array(samples_train_fold), np.array(labels_train_fold))
                samples_test_fold = get_samples_selected_features_from(config, samples_test_fold, random_state,
                                                                       num_task + 1, feature_selection_technique)
                prediction = np.array(model.predict(samples_test_fold).tolist())
                labels_test_folds.append(labels_test_fold)
                if num_task == classification_tasks[0]:
                    results.append(prediction)
                else:
                    results = np.array(results)
                    results[index_test] = np.add(results[index_test], prediction)
            end = time.time()

            # Sets results into apposite variables
            if num_task == classification_tasks[0]:
                times_one_task += end - start
                for index_fold in range(0, len(results)):
                    performances_one_task = get_list_accuracy_from(performances_one_task, results[index_fold],
                                                                   labels_test_folds[index_fold], vote_to_overcame=1)
            elif num_task == classification_tasks[1]:
                times_two_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_two_tasks = get_list_accuracy_from(performances_two_tasks, results[index_fold],
                                                                    labels_test_folds[index_fold], vote_to_overcame=2)
            elif num_task == classification_tasks[2]:
                times_three_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_three_tasks = get_list_accuracy_from(performances_three_tasks, results[index_fold],
                                                                      labels_test_folds[index_fold], vote_to_overcame=2)
            elif num_task == classification_tasks[3]:
                times_four_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_four_tasks = get_list_accuracy_from(performances_four_tasks, results[index_fold],
                                                                     labels_test_folds[index_fold], vote_to_overcame=3)
            elif num_task == classification_tasks[4]:
                times_five_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_five_tasks = get_list_accuracy_from(performances_five_tasks, results[index_fold],
                                                                     labels_test_folds[index_fold], vote_to_overcame=3)
            elif num_task == classification_tasks[5]:
                times_six_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_six_tasks = get_list_accuracy_from(performances_six_tasks, results[index_fold],
                                                                    labels_test_folds[index_fold], vote_to_overcame=4)
            elif num_task == classification_tasks[6]:
                times_seven_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_seven_tasks = get_list_accuracy_from(performances_seven_tasks, results[index_fold],
                                                                      labels_test_folds[index_fold], vote_to_overcame=4)
            elif num_task == classification_tasks[7]:
                times_eight_tasks += end - start
                for index_fold in range(0, len(results)):
                    performances_eight_tasks = get_list_accuracy_from(performances_eight_tasks, results[index_fold],
                                                                      labels_test_folds[index_fold])

    # Refines results into apposite dictionary
    table_performances = clean_table_performances_ensemble_learning()
    performances_one_task.append(times_one_task / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_one_task)
    performances_two_tasks.append(times_two_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_two_tasks)
    performances_three_tasks.append(times_three_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_three_tasks)
    performances_four_tasks.append(times_four_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_four_tasks)
    performances_five_tasks.append(times_five_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_five_tasks)
    performances_six_tasks.append(times_six_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_six_tasks)
    performances_seven_tasks.append(times_seven_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_seven_tasks)
    performances_eight_tasks.append(times_eight_tasks / repetitions)
    table_performances = update_table_experiment_from(table_performances, performances_eight_tasks)

    # Saves results
    write_csv_in(get_path_performances_ensemble_learning_default_hyperparameters(config) + "_random_state_" +
                 str(random_state) + ".csv", table_performances)
