"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file baseline_experiment_best_hyperparameters.py
PURPOSE OF THE FILE: Performs Baseline experiments with the best hyperparameters.
"""

import os

from tqdm import tqdm

from utils.constants import models_info
from utils.feature_selection import feature_selection
from utils.models_and_hyperparameters import get_best_models_all_features, get_best_models_selected_features
from utils.utilities import write_csv_in, perform_experiment_all_features_from, \
    perform_experiment_selected_features_from, clean_table_performances, get_path_baseline_selected_features


def get_path_performances_baseline_best_hyperparameters(config):
    """Returns path where (baseline) performances of models using the best hyperparameters has to be saved

    Args:
        config (dict): The instance of config.yml

    Return:
        string: The path where (baseline) performances of models using the best hyperparameters has to be saved
    """
    return config['baseline_best_hyperparameters']['performances']


def baseline_experiment_best_hyperparameters(config, samples, labels, random_state):
    """Performs models' performances of the baseline using the best hyperparameters and save them into a csv

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
    """
    # Fetches selected features
    if not os.path.exists(get_path_baseline_selected_features(config) + "_random_state_" + str(random_state)):
        print("Selected features on all task not detected."
              "\nPerforming Feature Selection Techniques...")
        feature_selection(config, samples, labels, random_state)

    # Fetches configured models
    table_performances = clean_table_performances()
    all_features_configurations = get_best_models_all_features(config, samples, labels, random_state)
    datasets, feature_selection_configurations = get_best_models_selected_features(config, samples, labels,
                                                                                   random_state)

    # Trains configurations Model/Feature-Selection-Technique
    for model_info in tqdm(models_info, "Training all best models on all tasks (Baseline Experiment)"):
        name_model = model_info[0]
        table_performances = perform_experiment_all_features_from(all_features_configurations, samples, labels,
                                                                  name_model,
                                                                  random_state, table_performances)
        table_performances = perform_experiment_selected_features_from(feature_selection_configurations, datasets,
                                                                       labels, name_model,
                                                                       random_state, table_performances)

    # Saves results
    write_csv_in(
        get_path_performances_baseline_best_hyperparameters(config) + "_random_state_" + str(random_state) + ".csv",
        table_performances)
