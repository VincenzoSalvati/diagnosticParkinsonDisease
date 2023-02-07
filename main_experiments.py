"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file main_experiments.py
PURPOSE OF THE FILE: Fetches dataset and starts experiments.
"""

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from experiments.baseline_experiment import baseline_experiment
from experiments.baseline_experiment_best_hyperparameters import baseline_experiment_best_hyperparameters
from experiments.tasks_experiment import tasks_experiment
from experiments.tasks_experiment_best_hyperparameters import tasks_experiment_best_hyperparameters
from experiments.tasks_final_ensemble_learning_experiment import tasks_final_ensemble_learning_experiment
from experiments.tasks_final_ensemble_learning_experiment_best_hyperparameters import \
    tasks_final_ensemble_learning_experiment_best_hyperparameters
from utils.constants import random_states
from utils.utilities import read_csv_from


def get_path_dataset(config):
    """Returns path where csv of all tasks is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of csv containing all tasks into a unique file
    """
    return config['features']['dataset']['all_features']


def get_path_dataset_tasks(config):
    """Returns path where csvs of tasks are contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of csv containing tasks in different files
    """
    return config['features']['dataset_tasks']['all_features']


def fetch_dataset(config, num_task=None):
    """Extracts data from PaHaW dataset

    Args:
        config (dict): The instance of config.yml
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        np.array[float]: Features
        np.array[]: Labels
    """

    def invert_labels(labels):
        """Invert 0 to 1 labels and vice versa from a dataset

        Args:
            labels (List[]): The label for each sample

        Return:
            List[]: The inverted labels' classes
        """
        labels_to_return = []
        for label in labels:
            if label == 0:
                labels_to_return.append(1)
            else:
                labels_to_return.append(0)
        return labels_to_return

    if num_task is None:
        dataset = read_csv_from(get_path_dataset(config))
    else:
        dataset = read_csv_from(get_path_dataset_tasks(config) + str(num_task) + ".csv")

    # Sets the normalisation samples with a mean equal to zero and a unit variance
    scaler = StandardScaler()
    scaler.fit(np.array(dataset[:, 0:-1], float))
    return scaler.transform(np.array(dataset[:, 0:-1], float)), np.array(invert_labels(dataset[:, -1]))


if __name__ == '__main__':
    # Fetch data from dataset
    with open('utils/config.yml') as file:
        config = yaml.full_load(file)
    samples, labels = fetch_dataset(config)
    dataset_tasks = [fetch_dataset(config, num_task=1), fetch_dataset(config, num_task=2),
                     fetch_dataset(config, num_task=3), fetch_dataset(config, num_task=4),
                     fetch_dataset(config, num_task=5), fetch_dataset(config, num_task=6),
                     fetch_dataset(config, num_task=7), fetch_dataset(config, num_task=8)]

    # For each models' initial conditions
    for random_state in random_states:
        print("Random state in uso: " + str(random_state))

        # What experiments to do
        flag_baseline_experiment = False
        flag_baseline_best_experiment = False
        flag_tasks_experiment = False
        flag_tasks_best_experiment = False
        flag_tasks_ensemble_learning_experiment = False
        flag_tasks_ensemble_learning_best_experiment = False

        while True:
            answer = input(
                'Do you want to perform the baseline experiment with default hyperparameters? '
                '[Y/N]. ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_baseline_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_baseline_experiment = False
                break

        while True:
            answer = input(
                'Do you want to run the baseline experiment with the best hyperparameters? '
                '[Y/N] ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_baseline_best_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_baseline_best_experiment = False
                break

        while True:
            answer = input(
                'Do you want to perform the task experiment with default hyperparameters? '
                '[Y/N] ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_tasks_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_tasks_experiment = False
                break

        while True:
            answer = input(
                'Do you want to perform the task experiment with the best hyperparameters? '
                '[Y/N] ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_tasks_best_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_tasks_best_experiment = False
                break

        while True:
            answer = input(
                'Do you want to perform the ensemble learning experiment on tasks with default hyperparameters? '
                '[Y/N] ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_tasks_ensemble_learning_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_tasks_ensemble_learning_experiment = False
                break

        while True:
            answer = input(
                'Do you want to perform the ensemble learning experiment on the tasks with the best hyperparameters? '
                '[Y/N] ')
            if answer.lower() == "y" or answer.lower() == "yes":
                flag_tasks_ensemble_learning_best_experiment = True
                break
            elif answer.lower() == "n" or answer.lower() == "no":
                flag_tasks_ensemble_learning_best_experiment = False
                break

        # Performs chosen experiments
        if flag_baseline_experiment:
            baseline_experiment(config, samples, labels, random_state)
        if flag_baseline_best_experiment:
            baseline_experiment_best_hyperparameters(config, samples, labels, random_state)
        if flag_tasks_experiment:
            tasks_experiment(config, dataset_tasks, random_state)
        if flag_tasks_best_experiment:
            tasks_experiment_best_hyperparameters(config, dataset_tasks, random_state)
        if flag_tasks_ensemble_learning_experiment:
            tasks_final_ensemble_learning_experiment(config, dataset_tasks, random_state)
        if flag_tasks_ensemble_learning_best_experiment:
            tasks_final_ensemble_learning_experiment_best_hyperparameters(config, dataset_tasks, random_state)
