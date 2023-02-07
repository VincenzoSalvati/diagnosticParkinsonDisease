"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file feature_selection.py
PURPOSE OF THE FILE: Applies Feature Selection techniques.
"""

import operator
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.constants import selected_features_by_techniques
from utils.skfeature.function.similarity_based.reliefF import reliefF
from utils.skfeature.function.statistical_based.CFS import cfs
from utils.utilities import write_string_in, serialize_object_in, get_path_baseline_selected_features, \
    get_path_tasks_selected_features


def structure_txt_to_report_in(path, features):
    """Write selected features from each technique into a csv

    Args:
        path (string): Path of csv
        features (dict): Selected features from each technique
    """

    def get_structured_dictionary_to_report_from(copy_features, features, feature_selection_technique):
        """Returns selected features from each technique

        Args:
            copy_features (dict): Selected features from each technique
            features (dict): Selected features from each technique
            feature_selection_technique (string): The name of feature selection technique

        Return:
            dict: Selected features from each technique
        """
        copy_features[feature_selection_technique] = \
            {"features": features[feature_selection_technique]["features"],
             "num_features":
                 {"all_tasks": features[feature_selection_technique]["num_features"]["all_tasks"],
                  "task_1": features[feature_selection_technique]["num_features"]["task_1"],
                  "task_2": features[feature_selection_technique]["num_features"]["task_2"],
                  "task_3": features[feature_selection_technique]["num_features"]["task_3"],
                  "task_4": features[feature_selection_technique]["num_features"]["task_4"],
                  "task_5": features[feature_selection_technique]["num_features"]["task_5"],
                  "task_6": features[feature_selection_technique]["num_features"]["task_6"],
                  "task_7": features[feature_selection_technique]["num_features"]["task_7"],
                  "task_8": features[feature_selection_technique]["num_features"]["task_8"]},
             "time[s]": features[feature_selection_technique]["time[s]"]}
        return copy_features

    copy_features = features
    copy_features = get_structured_dictionary_to_report_from(copy_features, features, "RFE_SVM")
    copy_features = get_structured_dictionary_to_report_from(copy_features, features, "RFE_LR")
    copy_features = get_structured_dictionary_to_report_from(copy_features, features, "RFE_RF")
    write_string_in(path, str(copy_features))


def get_dict_of_selected_feature_from(features, selected_features, feature_selection_technique, elapsed_time, num_task,
                                      samples=None):
    """Returns selected features from each technique

    Args:
        features (dict): Selected features from each technique
        selected_features (List[]): Selected features from each technique
        feature_selection_technique (string): The name of feature selection technique
        elapsed_time (float): The elapsed time
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

        samples (int): Samples' selected features

    Return:
        dict: Selected features from each technique
    """

    def count_tasks_from(features, num_task):
        """Counts occurrences for each task

        Args:
            features (dict): Selected features from each technique
            num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

        Return:
            list[]: Occurrences for each task
        """
        count = {"task_1": 0, "task_2": 0, "task_3": 0, "task_4": 0, "task_5": 0, "task_6": 0, "task_7": 0, "task_8": 0}
        for feature in features:
            if num_task is None:
                if 0 <= feature < 423:
                    count["task_1"] += 1
                elif 423 <= feature < 423 * 2:
                    count["task_2"] += 1
                elif 423 * 2 <= feature < 423 * 3:
                    count["task_3"] += 1
                elif 423 * 3 <= feature < 423 * 4:
                    count["task_4"] += 1
                elif 423 * 4 <= feature < 423 * 5:
                    count["task_5"] += 1
                elif 423 * 5 <= feature < 423 * 6:
                    count["task_6"] += 1
                elif 423 * 6 <= feature < 423 * 7:
                    count["task_7"] += 1
                elif 423 * 7 <= feature < 423 * 8:
                    count["task_8"] += 1
            else:
                count["task_" + str(num_task)] += 1
        return count

    features[feature_selection_technique]["features"] = selected_features
    features[feature_selection_technique]["num_features"]["all_tasks"] = len(selected_features)
    count_tasks = count_tasks_from(selected_features, num_task)
    features[feature_selection_technique]["num_features"]["task_1"] = count_tasks["task_1"]
    features[feature_selection_technique]["num_features"]["task_2"] = count_tasks["task_2"]
    features[feature_selection_technique]["num_features"]["task_3"] = count_tasks["task_3"]
    features[feature_selection_technique]["num_features"]["task_4"] = count_tasks["task_4"]
    features[feature_selection_technique]["num_features"]["task_5"] = count_tasks["task_5"]
    features[feature_selection_technique]["num_features"]["task_6"] = count_tasks["task_6"]
    features[feature_selection_technique]["num_features"]["task_7"] = count_tasks["task_7"]
    features[feature_selection_technique]["num_features"]["task_8"] = count_tasks["task_8"]
    if feature_selection_technique.__contains__("RFE"):
        features[feature_selection_technique]["samples"] = samples
    features[feature_selection_technique]["time[s]"] = elapsed_time
    return features


def get_selected_features_from(samples, labels, random_state, num_task):
    """Returns selected features

    Args:
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

    Return:
        dict: Selected features
    """

    def cfs_from(samples, labels):
        """Returns features selected by CFS technique

        Args:
            samples (List[]): Features for each sample
            labels (List[]): The label for each sample

        Return:
            List[]: Columns of features selected by CFS technique
        """

        def remove_useless_value(cfs_features):
            """Remove useless value from columns of selected features

            Args:
                cfs_features (List[]): Columns of selected features

            Return:
                List[]: Columns of features selected by CFS technique
            """
            sorted_features_cfs = []
            for feature in cfs_features:
                if not feature == -1:
                    sorted_features_cfs.append(feature)
            return sorted_features_cfs

        return remove_useless_value(cfs(samples, labels))

    def relieff_from(samples, labels):
        """Returns features selected by ReliefF technique

        Args:
            samples (List[]): Features for each sample
            labels (List[]): The label for each sample

        Return:
            List[]: Columns of features selected by ReliefF technique
        """

        def get_descending_sort_from(relief_f_features):
            """Sorts scored features in a descending way

            Args:
                relief_f_features (List[(float,float)]): Scored features

            Return:
                List[(float,float)]: Sorted scored features
            """
            dict_sorted_scored_features = {}
            index_feature = 0
            for score in relief_f_features:
                dict_sorted_scored_features.__setitem__(index_feature, score)
                index_feature += 1
            dict_sorted_scored_features = dict(
                sorted(dict_sorted_scored_features.items(), key=operator.itemgetter(1), reverse=True))
            return dict_sorted_scored_features

        def get_relevant_feature_from(relief_f_features, score_to_overcome=0):
            """Removes features which score does not overcome the minimum one

            Args:
                relief_f_features (List[(float,float)]): Scored features
                score_to_overcome (int): The minimum score to overcome

            Return:
                List[]: Columns of features selected by ReliefF technique
            """
            scored_features_relief_f = []
            for item in relief_f_features.items():
                if not item[1] <= score_to_overcome:
                    scored_features_relief_f.append(item[0])
            return scored_features_relief_f

        dict_sorted_scored_features = get_descending_sort_from(reliefF(samples, labels))
        return get_relevant_feature_from(dict_sorted_scored_features)

    def recursive_feature_elimination_from(samples, labels, model):
        """Returns features selected by Recursive Feature Elimination technique

        Args:
            samples (List[]): Features for each sample
            labels (List[]): The label for each sample
            model (Class): The learning algorithm

        Return:
            List[][]: The dataset's samples which features selected by Recursive Feature Elimination technique
            List[]: Columns of features selected by Recursive Feature Elimination technique
        """

        def get_selected_features_from(original_dataset, edited_dataset):
            """Returns columns of features selected by Recursive Feature Elimination technique

            Args:
                original_dataset (list[][]): The original dataset's samples which contains all features
                edited_dataset (list[][]): The dataset's samples which useless features are removed

            Return:
                List[]: Columns of features selected by Recursive Feature Elimination technique
            """
            selected_feature = []
            for col_to_check in edited_dataset.T:
                col = 0
                for original_col in original_dataset.T:
                    if np.array_equal(original_col, col_to_check):
                        selected_feature.append(col)
                        break
                    col += 1
            return selected_feature

        rfe = RFE(estimator=model)
        rfe.fit(samples, labels)
        edited_dataset = rfe.transform(samples)
        selected_features = get_selected_features_from(samples, edited_dataset)
        return edited_dataset, selected_features

    def get_features_by_guided_random_forest_from(num_task):
        """Returns features selected by Guided Random Forest technique

        Args:
            num_task (int): The number of the task. If it is equal to None, it considers all tasks at once

        Return:
            List[]: Columns of features selected by Guided Random Forest technique
        """
        if num_task is None:
            return [574, 673, 839, 1102, 1126, 1484, 1748, 2835, 3343]
        elif num_task == 1:
            return [104, 158, 209, 233, 296, 305, 306, 308, 351, 377, 386, 399, 420]
        elif num_task == 2:
            return [30, 151, 175, 187, 294, 306, 416, 420]
        elif num_task == 3:
            return [82, 85, 256, 277, 280, 372, 397, 400, 402, 403, 408, 417]
        elif num_task == 4:
            return [88, 160, 175, 203, 212, 215, 217, 220, 241, 243, 288, 419]
        elif num_task == 5:
            return [18, 26, 67, 248, 264, 345, 417, 418, 420, 421]
        elif num_task == 6:
            return [36, 48, 124, 227, 252, 324, 374, 420]
        elif num_task == 7:
            return [107, 125, 168, 290, 346, 416, 418]
        elif num_task == 8:
            return [56, 131, 143, 270, 342, 373, 384, 394, 396, 408, 416]

    # Fills dictionary of selected features for each Feature Selection Technique
    selected_features = selected_features_by_techniques

    # CFS
    start_cfs = time.time()
    cfs_features = cfs_from(samples, labels)
    end_cfs = time.time()
    selected_features = get_dict_of_selected_feature_from(selected_features, cfs_features, "CFS", end_cfs - start_cfs,
                                                          num_task)

    # ReliefF
    start_relief_f = time.time()
    relief_f_features = relieff_from(samples, labels)
    end_relief_f = time.time()
    selected_features = get_dict_of_selected_feature_from(selected_features, relief_f_features, "ReliefF",
                                                          end_relief_f - start_relief_f, num_task)

    # RFE-SVM
    start_rfe_svm = time.time()
    rfe_svm_features = recursive_feature_elimination_from(samples, labels,
                                                          SVC(kernel="linear", random_state=random_state))
    end_rfe_svm = time.time()
    selected_features = get_dict_of_selected_feature_from(selected_features, rfe_svm_features[1], "RFE_SVM",
                                                          end_rfe_svm - start_rfe_svm, num_task, rfe_svm_features[0])

    # RFE-LR
    start_rfe_lr = time.time()
    rfe_lr_features = recursive_feature_elimination_from(samples, labels, LogisticRegression(random_state=random_state))
    end_rfe_lr = time.time()
    selected_features = get_dict_of_selected_feature_from(selected_features, rfe_lr_features[1], "RFE_LR",
                                                          end_rfe_lr - start_rfe_lr, num_task, rfe_lr_features[0])

    # RFE-RF
    start_rfe_rf = time.time()
    rfe_rf_features = recursive_feature_elimination_from(samples, labels,
                                                         RandomForestClassifier(random_state=random_state))
    end_rfe_rf = time.time()
    selected_features = get_dict_of_selected_feature_from(selected_features, rfe_rf_features[1], "RFE_RF",
                                                          end_rfe_rf - start_rfe_rf, num_task, rfe_rf_features[0])

    # GRF
    grf_features = get_features_by_guided_random_forest_from(num_task)
    grf_times_sec = [4.862, 0.596, 0.620, 0.604, 0.614, 0.640, 0.631, 0.640, 0.618]
    elapsed_time_grf = grf_times_sec[0] if num_task is None else grf_times_sec[num_task]
    selected_features = get_dict_of_selected_feature_from(selected_features, grf_features, "GRF", elapsed_time_grf,
                                                          num_task)
    return selected_features


def feature_selection(config, samples, labels, random_state, num_task=None):
    """Performs feature selection techniques

    Args:
        config (dict): The instance of config.yml
        samples (List[]): Features for each sample
        labels (List[]): The label for each sample
        random_state (int): The initial condition of each model (where is possible)
        num_task (int): The number of the task. If it is equal to None, it considers all tasks at once
    """
    # Fetches selected_features
    selected_features = get_selected_features_from(samples, labels, random_state, num_task)

    # Saves selected features
    if num_task is None:
        serialize_object_in(
            get_path_baseline_selected_features(config) + "_random_state_" + str(random_state), selected_features)
        structure_txt_to_report_in(
            get_path_baseline_selected_features(config) + "_random_state_" + str(random_state) + ".txt",
            selected_features)
    else:
        serialize_object_in(
            get_path_tasks_selected_features(config) + str(num_task) + "_random_state_" + str(random_state),
            selected_features)
        structure_txt_to_report_in(
            get_path_tasks_selected_features(config) + str(num_task) + "_random_state_" + str(random_state) + ".txt",
            selected_features)
