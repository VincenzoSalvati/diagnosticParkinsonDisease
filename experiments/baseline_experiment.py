"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file baseline_experiment.py
PURPOSE OF THE FILE: Performs Baseline experiments.
"""

import os

from tqdm import tqdm

from utils.constants import models_info
from utils.feature_selection import feature_selection
from utils.models_and_hyperparameters import get_models_selected_features, get_models_all_features
from utils.utilities import write_csv_in, perform_experiment_all_features_from, \
    perform_experiment_selected_features_from, clean_table_performances, get_path_baseline_selected_features


def get_path_performances_baseline_default_hyperparameters(config):
    """Returns path where (baseline) performances of models using default hyperparameters has to be saved

    Args:
        config (dict): The instance of config.yml

    Return:
        string: The path where (baseline) performances of models using default hyperparameters has to be saved
    """
    return config['baseline_default_hyperparameters']['performances']


def baseline_experiment(config, samples, labels, random_state):
    """Performs models' performances of the baseline using default hyperparameters and save them into a csv

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
    all_features_configurations = get_models_all_features(config, random_state)
    datasets, feature_selection_configurations = get_models_selected_features(config, samples, random_state)

    # Trains configurations Model/Feature-Selection-Technique
    for model_info in tqdm(models_info, "Training all models on all tasks (Baseline Experiment)"):
        name_model = model_info[0]
        table_performances = perform_experiment_all_features_from(all_features_configurations, samples, labels,
                                                                  name_model,
                                                                  random_state, table_performances)
        table_performances = perform_experiment_selected_features_from(feature_selection_configurations, datasets,
                                                                       labels, name_model,
                                                                       random_state, table_performances)

    # Saves results
    write_csv_in(
        get_path_performances_baseline_default_hyperparameters(config) + "_random_state_" + str(random_state) + ".csv",
        table_performances)
