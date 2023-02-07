"""
Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2022/2023

Supervisor:
Marcelli      Angelo      amarcelli@unisa.it

Candidate:
Salvati       Vincenzo    v.salvati10@studenti.unisa.it      0622701550

@file constants.py
PURPOSE OF THE FILE: Provides useful constants to use in different files.py.
"""

# Models' initial conditions (where is possible)

random_states = [0, 1, 8, 42, 47]

# K Fold Stratified Cross-Validation

repetitions = 5
folds = 3

# Models and Hyperparameters

hyperparameters_knn = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

hyperparameters_svm = {'kernel': ('linear', 'rbf'),
                       'C': [pow(2, -8), pow(2, -7), pow(2, -6), pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2),
                             pow(2, -1),
                             pow(2, 0),
                             pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4), pow(2, 5), pow(2, 6), pow(2, 7), pow(2, 8)],
                       'gamma': [pow(2, -9), pow(2, -8), pow(2, -7), pow(2, -6), pow(2, -5),
                                 pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1),
                                 pow(2, 0),
                                 pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4),
                                 pow(2, 5), pow(2, 6), pow(2, 7), pow(2, 8), pow(2, 9)]}
hyperparameters_lr = {'C': [0.005 * i for i in range(1001)]}

hyperparameters_rf = {'n_estimators': [100, 200, 500, 1000, 2000],
                      'max_depth': [10, 20, 50, 100, None],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}

models_info = [("K-NN", hyperparameters_knn),
               ("SVM", hyperparameters_svm),
               ("LR", hyperparameters_lr),
               ("RF", hyperparameters_rf)]

# Feature Selection

selected_features_by_techniques = {"CFS":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "time[s]": -1},
                                   "ReliefF":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "time[s]": -1},
                                   "RFE_SVM":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "samples": None,
                                        "time[s]": -1},
                                   "RFE_LR":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "samples": None,
                                        "time[s]": -1},
                                   "RFE_RF":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "samples": None,
                                        "time[s]": -1},
                                   "GRF":
                                       {"features": [],
                                        "num_features":
                                            {"all_tasks": 0,
                                             "task_1": 0,
                                             "task_2": 0,
                                             "task_3": 0,
                                             "task_4": 0,
                                             "task_5": 0,
                                             "task_6": 0,
                                             "task_7": 0,
                                             "task_8": 0},
                                        "time[s]": -1}}

# Baseline and Tasks experiments

all_features_configurations = {"K-NN":
                                   {"hyperparameters":
                                        {"n_neighbors": 5},
                                    "best_accuracy": -1},
                               "SVM":
                                   {"hyperparameters":
                                        {"kernel": "rbf",
                                         "C": 1,
                                         "gamma": "scale"},
                                    "best_accuracy": -1},
                               "LR":
                                   {"hyperparameters":
                                        {"C": 1},
                                    "best_accuracy": -1},
                               "RF":
                                   {"hyperparameters":
                                        {"n_estimators": 100,
                                         "max_depth": None,
                                         "min_samples_split": 2,
                                         "min_samples_leaf": 1},
                                    "best_accuracy": -1}}

feature_selection_configurations = {"K-NN":
                                        {"CFS":
                                             {"hyperparameters":
                                                  {"n_neighbors": 5},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "ReliefF":
                                             {"hyperparameters":
                                                  {"n_neighbors": 5},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "GRF":
                                             {"hyperparameters":
                                                  {"n_neighbors": 5},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1}},
                                    "SVM":
                                        {"CFS":
                                             {"hyperparameters":
                                                  {"kernel": "rbf",
                                                   "C": 1,
                                                   "gamma": "scale"},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "ReliefF":
                                             {"hyperparameters":
                                                  {"kernel": "rbf",
                                                   "C": 1,
                                                   "gamma": "scale"},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "RFE":
                                             {"hyperparameters":
                                                  {"kernel": "rbf",
                                                   "C": 1,
                                                   "gamma": "scale"},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "GRF":
                                             {"hyperparameters":
                                                  {"kernel": "rbf",
                                                   "C": 1,
                                                   "gamma": "scale"},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1}},
                                    "LR":
                                        {"CFS":
                                             {"hyperparameters":
                                                  {"C": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "ReliefF":
                                             {"hyperparameters":
                                                  {"C": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "RFE":
                                             {"hyperparameters":
                                                  {"n_estimators": -1,
                                                   "C": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "GRF":
                                             {"hyperparameters":
                                                  {"C": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1}},
                                    "RF":
                                        {"CFS":
                                             {"hyperparameters":
                                                  {"n_estimators": 100,
                                                   "max_depth": None,
                                                   "min_samples_split": 2,
                                                   "min_samples_leaf": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "ReliefF":
                                             {"hyperparameters":
                                                  {"n_estimators": 100,
                                                   "max_depth": None,
                                                   "min_samples_split": 2,
                                                   "min_samples_leaf": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "RFE":
                                             {"hyperparameters":
                                                  {"n_estimators": 100,
                                                   "max_depth": None,
                                                   "min_samples_split": 2,
                                                   "min_samples_leaf": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1},
                                         "GRF":
                                             {"hyperparameters":
                                                  {"n_estimators": 100,
                                                   "max_depth": None,
                                                   "min_samples_split": 2,
                                                   "min_samples_leaf": 1},
                                              "best_accuracy": -1,
                                              "num_selected_features": -1}}}

table_performances = {
    "models": ["K-NN", "K-NN", "K-NN", "K-NN", "K-NN",
               "SVM", "SVM", "SVM", "SVM", "SVM",
               "LR", "LR", "LR", "LR", "LR",
               "RF", "RF", "RF", "RF", "RF"],
    "features": ["all_features", "CFS", "ReliefF", "RFE", "GRF",
                 "all_features", "CFS", "ReliefF", "RFE", "GRF",
                 "all_features", "CFS", "ReliefF", "RFE", "GRF",
                 "all_features", "CFS", "ReliefF", "RFE", "GRF"],
    "min_accuracies[%]": [],
    "max_accuracies[%]": [],
    "mean_accuracies[%]": [],
    "sdv_accuracies[%]": [],
    "time[s]": []
}

# Ensemble Learning Experiment

classification_tasks = [2, 1, 3, 0, 7, 6, 5, 4]

ensemble_learning_configurations = {
    "task_1":
        {"model": "LR",
         "feature_selection_technique": "GRF"},
    "task_2":
        {"model": "LR",
         "feature_selection_technique": "GRF"},
    "task_3":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_4":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_5":
        {"model": "LR",
         "feature_selection_technique": "GRF"},
    "task_6":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_7":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_8":
        {"model": "RF",
         "feature_selection_technique": "GRF"}
}

ensemble_learning_configurations_best_hyperparameters = {
    "task_1":
        {"model": "SVM",
         "feature_selection_technique": "GRF"},
    "task_2":
        {"model": "SVM",
         "feature_selection_technique": "GRF"},
    "task_3":
        {"model": "LR",
         "feature_selection_technique": "GRF"},
    "task_4":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_5":
        {"model": "LR",
         "feature_selection_technique": "RFE_LR"},
    "task_6":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_7":
        {"model": "RF",
         "feature_selection_technique": "GRF"},
    "task_8":
        {"model": "RF",
         "feature_selection_technique": "GRF"}
}

table_performances_ensemble_learning = {
    "combinations": ["RF-GRF (Task 3)",
                     "RF-GRF (Task 3), LR-GRF (Task 2)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4), LR-GRF (Task 1)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4), LR-GRF (Task 1), RF-GRF (Task 8)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4), LR-GRF (Task 1), RF-GRF (Task 8), "
                     "RF-GRF (Task 7)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4), LR-GRF (Task 1), RF-GRF (Task 8), "
                     "RF-GRF (Task 7), RF-GRF (Task 6)",
                     "RF-GRF (Task 3), LR-GRF (Task 2), RF-GRF (Task 4), LR-GRF (Task 1), RF-GRF (Task 8), "
                     "RF-GRF (Task 7), RF-GRF (Task 6), LR-GRF (Task 5)"],
    "min_accuracies[%]": [],
    "max_accuracies[%]": [],
    "mean_accuracies[%]": [],
    "sdv_accuracies[%]": [],
    "time[s]": []
}
