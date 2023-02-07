"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file main_demonstrator.py
PURPOSE OF THE FILE: Fetches dataset and starts demonstrator.
"""

import os

import numpy as np
import yaml
from sklearn.metrics import accuracy_score

from experiments.tasks_final_ensemble_learning_experiment_best_hyperparameters import get_model_from
from main_experiments import fetch_dataset
from utils.constants import classification_tasks, models_info
from utils.feature_selection import feature_selection
from utils.models_and_hyperparameters import get_models_selected_features, get_models_all_features, \
    get_samples_selected_features_from
from utils.utilities import get_path_tasks_selected_features

ensemble_learning_configurations = {
    "task_1":
        {"model": "SVM",
         "feature_selection_technique": "ReliefF"},
    "task_2":
        {"model": "SVM",
         "feature_selection_technique": "ReliefF"},
    "task_3":
        {"model": "K-NN",
         "feature_selection_technique": "GRF"},
    "task_4":
        {"model": "LR",
         "feature_selection_technique": "ReliefF"},
    "task_5":
        {"model": "LR",
         "feature_selection_technique": "GRF"},
    "task_6":
        {"model": "LR",
         "feature_selection_technique": "RFE_LR"},
    "task_7":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_8":
        {"model": "K-NN",
         "feature_selection_technique": "ReliefF"}
}


def get_balanced_sets_from(samples, labels, size_test_set=0.33):
    """Generates balanced splitting of dataset

    Args:
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        size_test_set (float): The size test's percentage

    Return:
        Array[]: train samples for training
        Array[]: train labels  for training
        Array[]: test samples for testing
        Array[]: test labels  for testing
    """

    # Sets the index positions of the sample sparing the classes to the side
    positions = labels.argsort()
    samples = samples[positions[::-1]]
    labels = labels[positions[::-1]]

    # Generates balanced folds
    number_elements_test_set = int(len(labels) * size_test_set)
    number_elements_train_set = len(labels) - number_elements_test_set

    train_samples, train_labels, test_samples, test_labels = [], [], [], []
    index_forward = 0
    index_backward = -1

    # Test set
    for num_element in range(1, number_elements_test_set + 1):
        if num_element % 2 == 0:
            test_samples.append(samples[index_forward, :])
            test_labels.append(labels[index_forward])
            index_forward += 1
        else:
            test_samples.append((samples[index_backward, :]))
            test_labels.append(labels[index_backward])
            index_backward -= 1
        num_element += 1

    # Train set
    for num_element in range(1, number_elements_train_set + 1):
        if num_element % 2 == 0:
            train_samples.append(samples[index_forward, :])
            train_labels.append(labels[index_forward])
            index_forward += 1
        else:
            train_samples.append((samples[index_backward, :]))
            train_labels.append(labels[index_backward])
            index_backward -= 1
        num_element += 1

    return np.array(train_samples), np.array(train_labels), np.array(test_samples), np.array(test_labels)


def get_best_result_from(samples, labels, random_state=42, num_task=None):
    """Returns the best result from baseline

    Args:
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        List[]: The best result from baseline
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

    # Fetches configured models
    if num_task is None:
        all_features_configurations = get_models_all_features(config, random_state)
        datasets, feature_selection_configurations = get_models_selected_features(config, samples, random_state)
    else:
        all_features_configurations = get_models_all_features(config, random_state, num_task=num_task)
        datasets, feature_selection_configurations = get_models_selected_features(config, samples, random_state,
                                                                                  num_task=num_task)

    # For each models' configuration without Feature Selection technique
    train_samples, train_labels, test_samples, test_labels = get_balanced_sets_from(samples, labels)
    feature_selection_techniques = ["CFS", "ReliefF", "RFE", "GRF"]
    results_all_features = []
    for model_info in models_info:
        name_model = model_info[0]

        # Extracts configured models
        model = get_model_from(all_features_configurations, name_model, random_state)

        # Trains and tests baseline
        model = model.fit(np.array(train_samples), np.array(train_labels))
        prediction = np.array(model.predict(test_samples).tolist())
        result = round(accuracy_score(test_labels, prediction), 3)

        # Saves result
        results_all_features.append([result, name_model, "All features"])

    # For each models' configuration with Feature Selection technique
    best_result_fs_techniques = []
    for feature_selection_technique in feature_selection_techniques:
        result = 0.000
        best_result = [0, "", ""]
        for model_info in models_info:
            name_model = model_info[0]
            if not (name_model == "K-NN" and feature_selection_technique == "RFE"):
                # Extracts samples' selected features
                samples = get_dataset_from(datasets, name_model, feature_selection_technique)
                train_samples, train_labels, test_samples, test_labels = get_balanced_sets_from(samples, labels)

                # Extracts configured models
                model = get_model_from(feature_selection_configurations, name_model, random_state,
                                       feature_selection_technique)

                # Trains and tests baseline
                model = model.fit(np.array(train_samples), np.array(train_labels))
                prediction = np.array(model.predict(test_samples).tolist())
                result = round(accuracy_score(test_labels, prediction), 3)

            # Check the best result
            if result > best_result[0]:
                best_result[0] = result
                best_result[1] = name_model
                best_result[2] = feature_selection_technique

        # Saves result
        best_result_fs_techniques.append(best_result)

    return results_all_features, best_result_fs_techniques


def get_best_from_ensemble(dataset_tasks, random_state=42):
    """Returns the best result from BFT

    Args:
        dataset_tasks (List[][]): Features and labels for each sample for each task
        random_state (int): The initial condition of each model (where is possible)

    Return:
        List[]: The best result from BFT
    """

    def get_accuracy_from(predictions, test, vote_to_overcame=5):
        """Performs definitive prediction from Ensemble Learning composed by specific experts

        Args:
            predictions (List[]): Experts' votes
            test (List[]): The test's labels
            vote_to_overcame (int): The majority vote to overcame for confirm the prediction

        Return:
            float: Accuracy achieved by specific experts
        """
        majority_voting_results = [1 if vote >= vote_to_overcame else 0 for vote in predictions]
        return round(accuracy_score(test, majority_voting_results), 3)

    # For each expert (from the best one until the worst one)
    performances = [(), (), (), (), (), (), (), ()]
    index_performances = 0
    predictions = []
    for num_task in classification_tasks[0:8]:
        samples, labels = np.array(dataset_tasks[num_task][0]), np.array(dataset_tasks[num_task][1])

        # Fetches selected features for the specific task and models' initial conditions
        if not os.path.exists(
                get_path_tasks_selected_features(config) + str(num_task + 1) + "_random_state_" + str(random_state)):
            print("Selected features on task " + str(num_task + 1) + " not detected." +
                  "\nPerforming Feature Selection Techniques...")
            feature_selection(config, samples, labels, random_state, num_task + 1)
        _, feature_selection_configurations = get_models_selected_features(config, samples, random_state,
                                                                           num_task=num_task + 1)

        # Fetches the best configuration Model/Feature-Selection-Technique for the specific task
        feature_selection_technique = ensemble_learning_configurations["task_" + str(num_task + 1)][
            "feature_selection_technique"]
        name_model = ensemble_learning_configurations["task_" + str(num_task + 1)]["model"]
        model = get_model_from(feature_selection_configurations, name_model, random_state,
                               feature_selection_technique)

        # Trains and tests BFT
        samples = get_samples_selected_features_from(config, samples, random_state, num_task + 1,
                                                     feature_selection_technique)
        train_samples, train_labels, test_samples, test_labels = get_balanced_sets_from(samples, labels)

        model = model.fit(np.array(train_samples), np.array(train_labels))
        prediction = np.array(model.predict(test_samples).tolist())
        if num_task == classification_tasks[0]:
            predictions = prediction
        else:
            predictions = np.add(predictions, prediction)

        # Sets results
        if num_task == classification_tasks[0]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=1),
                                                "task_" + str(classification_tasks[0] + 1))
        elif num_task == classification_tasks[1]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=2),
                                                "add task_" + str(classification_tasks[1] + 1))
        elif num_task == classification_tasks[2]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=2),
                                                "add task_" + str(classification_tasks[2] + 1))
        elif num_task == classification_tasks[3]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=3),
                                                "add task_" + str(classification_tasks[3] + 1))
        elif num_task == classification_tasks[4]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=3),
                                                "add task_" + str(classification_tasks[4] + 1))
        elif num_task == classification_tasks[5]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=4),
                                                "add task_" + str(classification_tasks[5] + 1))
        elif num_task == classification_tasks[6]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels, vote_to_overcame=4),
                                                "add task_" + str(classification_tasks[6] + 1))
        elif num_task == classification_tasks[7]:
            performances[index_performances] = (get_accuracy_from(predictions, test_labels),
                                                "add task_" + str(classification_tasks[7] + 1))
        index_performances += 1
    return performances


def get_max_result(results):
    """Returns maximum result

    Args:
        results (List[]): Acheaved results

    Return:
        list[]: The best result
    """
    max_result = [0.000, "", ""]
    for result in results:
        if result[0] > max_result[0]:
            max_result = result
    return max_result


if __name__ == '__main__':
    # Fetch data from dataset
    with open('utils/config.yml') as file:
        config = yaml.full_load(file)
    samples, labels = fetch_dataset(config)
    dataset_tasks = [fetch_dataset(config, num_task=1), fetch_dataset(config, num_task=2),
                     fetch_dataset(config, num_task=3), fetch_dataset(config, num_task=4),
                     fetch_dataset(config, num_task=5), fetch_dataset(config, num_task=6),
                     fetch_dataset(config, num_task=7), fetch_dataset(config, num_task=8)]

    # Baseline
    results_all_features, best_result_fs_techniques = get_best_result_from(samples, labels)
    print("\nBest result Baseline:")
    print("- No Feature Selection:\t", get_max_result(results_all_features))
    print("- Feature Selection:\t", get_max_result(best_result_fs_techniques))

    # Tasks
    print("\nBest result Tasks:")
    for num_task in range(0, 8):
        samples, labels = dataset_tasks[num_task]
        results_all_features, best_result_fs_techniques = get_best_result_from(samples, labels, num_task=num_task + 1)
        print(
            "Task " + str(num_task + 1) + ":\t" + str(get_max_result(results_all_features + best_result_fs_techniques)))

    # Ensemble Learning
    print("\nResult Ensemble Learning:\n", np.array(get_best_from_ensemble(dataset_tasks)))
