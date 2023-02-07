"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file models_and_hyperparameters.py
PURPOSE OF THE FILE: Fetches configuration Model/Feature-Selection-Techniques with hyperparameters.
"""

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from utils.constants import feature_selection_configurations, all_features_configurations, models_info
from utils.utilities import deserialize_object_from, write_string_in, serialize_object_in, \
    get_path_baseline_selected_features, get_path_tasks_selected_features


def get_path_default_hyperparameters_baseline_all_features(config):
    """Returns path where object of models' default hyperparameters (baseline) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' default hyperparameters (baseline)
    """
    return config['baseline_default_hyperparameters']['all_features_hyperparameters']


def get_path_best_hyperparameters_baseline_all_features(config):
    """Returns path where object of models' the best hyperparameters (baseline) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' the best hyperparameters (baseline)
    """
    return config['baseline_best_hyperparameters']['all_features_hyperparameters']


def get_path_default_hyperparameters_baseline_selected_features(config):
    """Returns path where object of models' default hyperparameters for selected feature (baseline) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' default hyperparameters for selected feature (baseline)
    """
    return config['baseline_default_hyperparameters']['selected_features_hyperparameters']


def get_path_best_hyperparameters_baseline_selected_features(config):
    """Returns path where object of models' the best hyperparameters for selected features (baseline) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' the best hyperparameters for selected features (baseline)
    """
    return config['baseline_best_hyperparameters']['selected_features_hyperparameters']


def get_path_default_hyperparameters_tasks_all_features(config):
    """Returns path where object of models' default hyperparameters (tasks) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' default hyperparameters (tasks)
    """
    return config['tasks_default_hyperparameters']['all_features_hyperparameters']


def get_path_best_hyperparameters_tasks_all_features(config):
    """Returns path where object of models' the best hyperparameters (tasks) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' the best hyperparameters (tasks)
    """
    return config['tasks_best_hyperparameters']['all_features_hyperparameters']


def get_path_default_hyperparameters_tasks_selected_features(config):
    """Returns path where object of models' default hyperparameters for selected feature (tasks) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' default hyperparameters for selected feature (tasks)
    """
    return config['tasks_default_hyperparameters']['selected_features_hyperparameters']


def get_path_best_hyperparameters_tasks_selected_features(config):
    """Returns path where object of models' the best hyperparameters for selected features (tasks) is contained

    Args:
        config (dict): The instance of config.yml

    Return:
        string: Path of object of models' the best hyperparameters for selected features (tasks)
    """
    return config['tasks_best_hyperparameters']['selected_features_hyperparameters']


# noinspection PyTypeChecker
def get_selected_features(config, samples, random_state, num_task=None):
    """Returns samples' selected features

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        List[]: Samples' selected features
    """

    def get_features_by_filter_from(selected_features):
        """Returns columns of features selected by Filters techniques

        Args:
            selected_features (dict): Selected features

        Return:
            List[]: Columns of features selected by CFS technique
            List[]: Columns of features selected by ReliefF technique
        """
        return selected_features["CFS"]["features"], selected_features["ReliefF"]["features"]

    def get_features_by_wrapper_from(selected_features, name_model):
        """Returns samples' features selected by Recursive Feature Elimination technique

        Args:
            selected_features (dict): Selected features
            name_model (string): The name of the learning algorithm

        Return:
            List[][]: The dataset's samples which features selected by Recursive Feature Elimination technique
            List[]: Columns of features selected by Recursive Feature Elimination technique
        """
        if name_model == "SVM":
            return selected_features["RFE_SVM"]["features"], selected_features["RFE_SVM"]["samples"]
        elif name_model == "LR":
            return selected_features["RFE_LR"]["features"], selected_features["RFE_LR"]["samples"]
        elif name_model == "RF":
            return selected_features["RFE_RF"]["features"], selected_features["RFE_RF"]["samples"]

    def get_features_by_ensemble_from(selected_features):
        """Returns columns of features selected by Guided Random Forest techniques

        Args:
            selected_features (dict): Selected features

        Return:
            List[]: Columns of features selected by Guided Random Forest technique
        """
        return selected_features["GRF"]["features"]

    def get_edited_samples_from(samples, features_to_keep):
        """Returns samples' selected features

        Args:
            samples (List[]): Features for each sample
            features_to_keep (List[]): Columns of selected features

        Return:
            List[]: Samples' selected features by CFS
            List[]: Samples' selected features by ReliefF
            List[]: Samples' selected features by RFE-SVM
            List[]: Samples' selected features by RFE-LR
            List[]: Samples' selected features by RFE-RF
            List[]: Samples' selected features by GRF
        """
        feature_to_delete = []
        if not features_to_keep == []:
            for feature in range(samples.shape[1]):
                if feature not in features_to_keep:
                    feature_to_delete.append(feature)
        return np.delete(samples, feature_to_delete, 1)

    # Fetches selected features for each Feature Selection technique
    if num_task is None:
        selected_features = deserialize_object_from(
            get_path_baseline_selected_features(config) + "_random_state_" + str(random_state))
    else:
        selected_features = deserialize_object_from(
            get_path_tasks_selected_features(config) + str(num_task) + "_random_state_" + str(random_state))

    # Filter techniques
    features_cfs, features_relief_f = get_features_by_filter_from(selected_features)

    # Wrapper technique
    dataset_rfe_svm = get_features_by_wrapper_from(selected_features, "SVM")
    dataset_rfe_lr = get_features_by_wrapper_from(selected_features, "LR")
    datasets_rfe_rf = get_features_by_wrapper_from(selected_features, "RF")

    # Ensemble technique
    features_grf = get_features_by_ensemble_from(selected_features)
    return (features_cfs, get_edited_samples_from(samples, features_cfs)), \
        (features_relief_f, get_edited_samples_from(samples, features_relief_f)), \
        dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, \
        (features_grf, get_edited_samples_from(samples, features_grf))


# Baseline and Tasks experiments - Default Hyperparameters
def get_models_all_features(config, random_state, num_task=None):
    """Returns configurations of baseline

    Args:
        config (dict): The instance of config.yml
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        dict: The configurations of baseline
    """

    # Return models' hyperparameters if it is possible
    if num_task is None:
        if os.path.exists(
                get_path_default_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state)):
            return deserialize_object_from(
                get_path_default_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state))
    else:
        if os.path.exists(
                get_path_default_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
                str(random_state)):
            return deserialize_object_from(
                get_path_default_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
                str(random_state))

    # Saves models' hyperparameters
    if num_task is None:
        serialize_object_in(
            get_path_default_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state),
            all_features_configurations)
        write_string_in(
            get_path_default_hyperparameters_baseline_all_features(config) + "_random_state_" +
            str(random_state) + ".txt", all_features_configurations)
    else:
        serialize_object_in(
            get_path_default_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
            str(random_state), all_features_configurations)
        write_string_in(
            get_path_default_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
            str(random_state) + ".txt", all_features_configurations)
    return all_features_configurations


def get_models_selected_features(config, samples, random_state, num_task=None):
    """Returns configurations of selected features baseline

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        dict: The configurations of selected features baseline
    """
    # Return models' configurations if it is possible
    dataset_cfs, dataset_relief_f, \
        dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, \
        dataset_grf = get_selected_features(config, samples, random_state, num_task)
    if num_task is None:
        if os.path.exists(
                get_path_default_hyperparameters_baseline_selected_features(config) + "_random_state_" +
                str(random_state)):
            return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
                deserialize_object_from(
                    get_path_default_hyperparameters_baseline_selected_features(config) + "_random_state_" +
                    str(random_state))
    else:
        if os.path.exists(
                get_path_default_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
                str(random_state)):
            return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
                deserialize_object_from(
                    get_path_default_hyperparameters_tasks_selected_features(config) + str(num_task) +
                    "_random_state_" + str(random_state))

    # Fills models' configurations
    for model_info in models_info:
        name_model = model_info[0]
        feature_selection_configurations[name_model]["CFS"]["num_selected_features"] = len(dataset_cfs[0])
        feature_selection_configurations[name_model]["ReliefF"]["num_selected_features"] = len(dataset_relief_f[0])
        if name_model == "SVM":
            feature_selection_configurations[name_model]["RFE"]["num_selected_features"] = len(dataset_rfe_svm[0])
        elif name_model == "LR":
            feature_selection_configurations[name_model]["RFE"]["num_selected_features"] = len(dataset_rfe_lr[0])
        elif name_model == "RF":
            feature_selection_configurations[name_model]["RFE"]["num_selected_features"] = len(datasets_rfe_rf[0])
        feature_selection_configurations[name_model]["GRF"]["num_selected_features"] = len(dataset_grf[0])

    # Saves models' configurations
    if num_task is None:
        serialize_object_in(
            get_path_default_hyperparameters_baseline_selected_features(config) + "_random_state_" + str(random_state),
            feature_selection_configurations)
        write_string_in(get_path_default_hyperparameters_baseline_selected_features(config) + "_random_state_" +
                        str(random_state) + ".txt", feature_selection_configurations)
    else:
        serialize_object_in(
            get_path_default_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
            str(random_state), feature_selection_configurations)
        write_string_in(
            get_path_default_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
            str(random_state) + ".txt", feature_selection_configurations)
    return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
        feature_selection_configurations


# Baseline and Tasks experiments - The best Hyperparameters
def set_configurations_from(model, name_model, hyperparameters, samples, labels, feature_selection_technique=None,
                            selected_features=None):
    """Set the best hyperparameters into the configuration's dictionary

    Args:
        model (Class): The learning algorithm
        name_model (string): The name of the learning algorithm
        hyperparameters (list[]): The model's hyperparameters
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        feature_selection_technique (string): The name of feature selection technique
        selected_features (list[]): Columns of selected features
    """

    # noinspection PyUnresolvedReferences
    def find_best_hyperparameters_from(model, hyperparameters, samples, labels, folds=3):
        """Returns the best hyperparameters and the best accuracy of the model

        Args:
            model (Class): The learning algorithm
            hyperparameters (list[]): The model's hyperparameters
            samples (List[]): Features for each sample
            labels (List[]): The label for each sample
            folds (int): The number of folds

        Return:
            dict: The best hyperparameters and the best accuracy of the model
        """
        search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=folds)
        result = search.fit(samples, labels)
        best_accuracy = result.best_score_
        best_hyperparameter = result.best_params_
        return best_hyperparameter, best_accuracy

    def set_best_configurations_from(name_model, best_hyperparameter, best_accuracy, feature_selection_technique,
                                     selected_features):
        """Set the best hyperparameters into the configuration's dictionary

        Args:
            name_model (string): The name of the learning algorithm
            best_hyperparameter (list[]): The best model's hyperparameters
            best_accuracy (list[]): The best model's accuracy
            feature_selection_technique (string): The name of feature selection technique
            selected_features (list[]): Columns of selected features
        """
        # K-NN
        if name_model == "K-NN":
            if feature_selection_technique is not None:
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "n_neighbors"] = best_hyperparameter["n_neighbors"]
                feature_selection_configurations[name_model][feature_selection_technique]["best_accuracy"] = \
                    best_accuracy
                feature_selection_configurations[name_model][feature_selection_technique]["num_selected_features"] = \
                    len(selected_features)
            else:
                all_features_configurations[name_model]["hyperparameters"]["n_neighbors"] = \
                    best_hyperparameter["n_neighbors"]
                all_features_configurations[name_model]["best_accuracy"] = \
                    best_accuracy

        # SVM
        elif name_model == "SVM":
            if feature_selection_technique is not None:
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "kernel"] = best_hyperparameter["kernel"]
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"]["C"] = \
                    best_hyperparameter["C"]
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "gamma"] = best_hyperparameter["gamma"]
                feature_selection_configurations[name_model][feature_selection_technique]["best_accuracy"] = \
                    best_accuracy
                feature_selection_configurations[name_model][feature_selection_technique]["num_selected_features"] = \
                    len(selected_features)
            else:
                all_features_configurations[name_model]["hyperparameters"]["kernel"] = \
                    best_hyperparameter["kernel"]
                all_features_configurations[name_model]["hyperparameters"]["C"] = \
                    best_hyperparameter["C"]
                all_features_configurations[name_model]["hyperparameters"]["gamma"] = \
                    best_hyperparameter["gamma"]
                all_features_configurations[name_model]["best_accuracy"] = \
                    best_accuracy

        # LR
        elif name_model == "LR":
            if feature_selection_technique is not None:
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"]["C"] = \
                    best_hyperparameter["C"]
                feature_selection_configurations[name_model][feature_selection_technique]["best_accuracy"] = \
                    best_accuracy
                feature_selection_configurations[name_model][feature_selection_technique]["num_selected_features"] = \
                    len(selected_features)
            else:
                all_features_configurations[name_model]["hyperparameters"]["C"] = \
                    best_hyperparameter["C"]
                all_features_configurations[name_model]["best_accuracy"] = \
                    best_accuracy

        # RF
        elif name_model == "RF":
            if feature_selection_technique is not None:
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "n_estimators"] = best_hyperparameter["n_estimators"]
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "max_depth"] = best_hyperparameter["max_depth"]
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "min_samples_split"] = best_hyperparameter["min_samples_split"]
                feature_selection_configurations[name_model][feature_selection_technique]["hyperparameters"][
                    "min_samples_leaf"] = best_hyperparameter["min_samples_leaf"]
                feature_selection_configurations[name_model][feature_selection_technique]["best_accuracy"] = \
                    best_accuracy
                feature_selection_configurations[name_model][feature_selection_technique]["num_selected_features"] = \
                    len(selected_features)
            else:
                all_features_configurations[name_model]["hyperparameters"]["n_estimators"] = \
                    best_hyperparameter["n_estimators"]
                all_features_configurations[name_model]["hyperparameters"]["max_depth"] = \
                    best_hyperparameter["max_depth"]
                all_features_configurations[name_model]["hyperparameters"]["min_samples_split"] = \
                    best_hyperparameter["min_samples_split"]
                all_features_configurations[name_model]["hyperparameters"]["min_samples_leaf"] = \
                    best_hyperparameter["min_samples_leaf"]
                all_features_configurations[name_model]["best_accuracy"] = \
                    best_accuracy

    best_hyperparameter, best_accuracy = find_best_hyperparameters_from(model, hyperparameters, samples, labels)
    set_best_configurations_from(name_model, best_hyperparameter, best_accuracy, feature_selection_technique,
                                 selected_features)


def get_model_from(name_model, random_state):
    """Returns the learning algorithm

    Args:
        name_model (string): The name of the learning algorithm
        random_state (int): The initial condition of each model (where is possible)

    Return:
        string: The learning algorithm
    """
    if name_model == "K-NN":
        return KNeighborsClassifier()
    elif name_model == "SVM":
        return SVC(random_state=random_state)
    elif name_model == "LR":
        return LogisticRegression(random_state=random_state)
    elif name_model == "RF":
        return RandomForestClassifier(random_state=random_state)


def get_best_models_all_features(config, samples, labels, random_state, num_task=None):
    """Returns configurations of the best hyperparameters baseline

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        dict: The configurations of the best hyperparameters baseline
        """
    # Return the models' best hyperparameters if it is possible
    if num_task is None:
        if os.path.exists(
                get_path_best_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state)):
            return deserialize_object_from(
                get_path_best_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state))
    else:
        if os.path.exists(
                get_path_best_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
                str(random_state)):
            return deserialize_object_from(
                get_path_best_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
                str(random_state))

    # Seeks models' best hyperparameters
    string_to_show = "Searching best hyperparameter for each model trained by all features"
    string_to_show += " - task " + str(num_task) if num_task is not None else ""
    for model_info in tqdm(models_info, string_to_show):
        name_model = model_info[0]
        model = get_model_from(name_model, random_state)
        hyperparameters = model_info[1]
        set_configurations_from(model, name_model, hyperparameters, samples, labels)

    # Saves models' best hyperparameters
    if num_task is None:
        serialize_object_in(
            get_path_best_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state),
            all_features_configurations)
        write_string_in(
            get_path_best_hyperparameters_baseline_all_features(config) + "_random_state_" + str(random_state) + ".txt",
            all_features_configurations)
    else:
        serialize_object_in(
            get_path_best_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
            str(random_state), all_features_configurations)
        write_string_in(
            get_path_best_hyperparameters_tasks_all_features(config) + str(num_task) + "_random_state_" +
            str(random_state) + ".txt", all_features_configurations)
    return all_features_configurations


def get_best_models_selected_features(config, samples, labels, random_state, num_task=None):
    """Returns configurations of the best hyperparameters and selected features baseline

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        dict: The configurations of the best hyperparameters and selected features baseline
    """
    # Return models' best configurations if it is possible
    dataset_cfs, dataset_relief_f, \
        dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, \
        dataset_grf = get_selected_features(config, samples, random_state, num_task)
    if num_task is None:
        if os.path.exists(
                get_path_best_hyperparameters_baseline_selected_features(config) + "_random_state_" +
                str(random_state)):
            return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
                deserialize_object_from(
                    get_path_best_hyperparameters_baseline_selected_features(config) + "_random_state_" +
                    str(random_state))
    else:
        if os.path.exists(
                get_path_best_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
                str(random_state)):
            return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
                deserialize_object_from(
                    get_path_best_hyperparameters_tasks_selected_features(config) + str(num_task) +
                    "_random_state_" + str(random_state))

    # Seeks models' best hyperparameters
    string_to_show = "Searching best hyperparameter for each model trained by selected features"
    string_to_show += " - task " + str(num_task) if num_task is not None else ""
    for model_info in tqdm(models_info, string_to_show):
        name_model = model_info[0]
        model = get_model_from(name_model, random_state)
        hyperparameters = model_info[1]
        set_configurations_from(model, name_model, hyperparameters, dataset_cfs[1], labels,
                                feature_selection_technique="CFS",
                                selected_features=dataset_cfs[0])
        set_configurations_from(model, name_model, hyperparameters, dataset_relief_f[1], labels,
                                feature_selection_technique="ReliefF",
                                selected_features=dataset_relief_f[0])
        if name_model == "SVM":
            set_configurations_from(model, name_model, hyperparameters, dataset_rfe_svm[1], labels,
                                    feature_selection_technique="RFE",
                                    selected_features=dataset_rfe_svm[0])
        elif name_model == "LR":
            set_configurations_from(model, name_model, hyperparameters, dataset_rfe_lr[1], labels,
                                    feature_selection_technique="RFE",
                                    selected_features=dataset_rfe_lr[0])
        elif name_model == "RF":
            set_configurations_from(model, name_model, hyperparameters, datasets_rfe_rf[1], labels,
                                    feature_selection_technique="RFE",
                                    selected_features=datasets_rfe_rf[0])
        set_configurations_from(model, name_model, hyperparameters, dataset_grf[1], labels,
                                feature_selection_technique="GRF",
                                selected_features=dataset_grf[0])

    # Saves models' best configurations
    if num_task is None:
        serialize_object_in(
            get_path_best_hyperparameters_baseline_selected_features(config) + "_random_state_" + str(random_state),
            feature_selection_configurations)
        write_string_in(
            get_path_best_hyperparameters_baseline_selected_features(config) + "_random_state_" +
            str(random_state) + ".txt", feature_selection_configurations)
    else:
        serialize_object_in(
            get_path_best_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
            str(random_state), feature_selection_configurations)
        write_string_in(
            get_path_best_hyperparameters_tasks_selected_features(config) + str(num_task) + "_random_state_" +
            str(random_state) + ".txt", feature_selection_configurations)
    return [dataset_cfs, dataset_relief_f, dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, dataset_grf], \
        feature_selection_configurations


# Ensemble Learning
def get_samples_selected_features_from(config, samples, random_state, num_task, feature_selection_technique):
    """Returns samples' selected feature

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once
        feature_selection_technique (string): The name of feature selection technique

    Return:
        List[]: Samples' selected feature
    """
    dataset_cfs, dataset_relief_f, \
        dataset_rfe_svm, dataset_rfe_lr, datasets_rfe_rf, \
        dataset_grf = get_selected_features(config, samples, random_state, num_task)
    if feature_selection_technique == "CFS":
        return dataset_cfs[1]
    elif feature_selection_technique == "ReliefF":
        return dataset_relief_f[1]
    elif feature_selection_technique == "RFE_SVM":
        return dataset_rfe_svm[1]
    elif feature_selection_technique == "RFE_LR":
        return dataset_rfe_lr[1]
    elif feature_selection_technique == "RFE_RF":
        return datasets_rfe_rf[1]
    elif feature_selection_technique == "GRF":
        return dataset_grf[1]
