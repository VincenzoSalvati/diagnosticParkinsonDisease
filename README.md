# Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2021/2022

## Candidate

Vincenzo Salvati - v.salvati10@studenti.unisa.it - 0622701550

# Abstract

The aim of this work concerns the study of the distinctive traits that better define the behaviour of Parkinsonians regarding their automatic motor mechanisms concerning their handwriting.
The approach intended to follow is related to the analysis of the writing tasks performed by each of the subjects, both healthy and affected by the disease, through some of the Feature Selection techniques for each of its categories and the most widely used Machine Learning models.
PaHaW is the dataset examined, which presents a total of 75 samples and it is very difficult to provide a fair representation of the reality of this problem, a further reason to focus only on specific aspects which lead to significant improvements in the accuracy of predictions. In this regard, the analysis tries to exploit the 8 tasks in question, whose raw features are inherent to the kinematics and dynamics, also through the use of statistical features that consider medians, differences between percentiles, standard deviations, correlations and so on.
The results that emerge from this study serve to consolidate the now already ascertained resoluteness of the models with the use of a vector containing all features of each task (baseline) concerning the use of single tasks, which in themselves they do not produce enough accuracy to be compared with the baseline. At the same time, what turns out is the effectiveness of FS techniques which considerably increase the performance reducing the computational load.
The second experiment allows us to identify those tasks that give more satisfaction than others and also about the statistical stability deriving from the Wilcoxon test as the composition of the dataset, the initial conditions of the learning models and the hyper-parameters vary.
The third and last experiment, on the other hand, is looking for the combination of the best tasks through Ensemble Learning that follows the Majority Voting approach among the best experts. What emerges from the last experiment is the deterioration of performance with the aid of tasks which use words. On the contrary, the most relevant tasks refer to the single letters, bigrams and trigrams which bring out the aspects captured by the degeneration of the automatic motor programs of subjects affected by Parkinson's, rather than healthy ones.
Furthermore, the results show evident goodness of correlated features through Pearson’s coefficient between pressure and acceleration and between pressure and jerk, especially along the horizontal axis because writing would require a greater extension in the amplitude of the wrist. To support what is said above, it has to be considered the control over the drafting of letters such as the "l" and, more pronouncedly, the "e" which require such stability that is determined only by an automatism of the movement acquired during life (akinesia).
Moreover, speed and the consequent drafting time of a subject who is in continuous relearning of the
motor movements linked to the drafting of sequences of characters play a quite decisive role in this
problem (bradykinesia).
Finally, what is assumed from these experiments is that the word tasks perform worse than the others
because, although there is no motor anticipation between letters in a Parkinsonian, they are
characterized by looping letters such as an "o" or a "p" which is difficult to practice even by a healthy
subject and of which there are no automatisms to perfectly draft letters.

## Paths

```.
|-- diagnosticParkinsonDisease
|   |-- experiments
|   |   |-- baseline_experiment.py
|   |   |-- baseline_experiment_best_hyperparameters.py
|   |   |-- tasks_experiment.py
|   |   |-- tasks_experiment_best_hyperparameters.py
|   |   |-- tasks_final_ensemble_learning_experiment.py
|   |   |-- tasks_final_ensemble_learning_experiment_best_hyperparameters.py
|   |-- files
|   |   |-- baseline_experiment
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- baseline_experiment_best_hyperparameters
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- features
|   |   |   |-- dataset
|   |   |   |-- dataset_Task
|   |   |-- tasks_experiment
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- tasks_experiment_best_hyperparameters
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- tasks_final_ensemble_learning_experiment
|   |   |-- tasks_final_ensemble_learning_experiment_best_hyperparameters
|   |-- utils
|   |   |-- skfeature
|   |   |-- config.yml
|   |   |-- constants.py
|   |   |-- feature_selection.py
|   |   |-- models_and_hyperparameters.py
|   |   |-- utilities.py
|   |-- main_demonstrator.py
|   |-- main_experiments.py
```

The developed code is divided in the following folders/files:

- The “experiments” folder contains all the files relative to the experiments. In particular, it contains:
    - baseline_experiment.py performs Baseline experiments
    - baseline_experiment_best_hyperparameters.py performs Baseline experiments with the best hyperparameters
    - tasks_experiment.py performs Tasks experiments
    - tasks_experiment_best_hyperparameters.py performs Tasks experiments with the best hyperparameters
    - tasks_final_ensemble_learning_experiment.py performs Ensemble Learning experiments
    - tasks_final_ensemble_learning_experiment_best_hyperparameters.py performs Ensemble Learning experiments with the
      best hyperparameters
- The “files” folder contains dataset PaHaW, models' hyperparameters and experiments' performances. In particular, it
  contains:
    - baseline_experiment folder contains hyperparameters used for the training of Baseline experiment and it lies on
      all features and selected features. Indeed, it contains also relative performances obtained by the models'
      configurations
    - baseline_experiment_best_hyperparameters folder contains the best hyperparameters used for the training of
      Baseline experiment and it lies on all features and selected features. Indeed, it contains also relative
      performances obtained by the models' configurations
    - features folder contains the dataset PaHaW with all the features at once and with all tasks it is composite of
    - tasks_experiment folder contains hyperparameters used for the training of Tasks experiment and it lies on all
      features and selected features. Indeed, it contains also relative performances obtained by the models'
      configurations
    - tasks_experiment_best_hyperparameters folder contains the best hyperparameters used for the training of Tasks
      experiment and it lies on all features and selected features. Indeed, it contains also relative performances
      obtained by the models' configurations
    - tasks_final_ensemble_learning_experiment folder contains performances obtained by the experts' combination of
      Ensemble Learning experiment
    - tasks_final_ensemble_learning_experiment_best_hyperparameters folder contains the best performances obtained by
      the experts' combination of Ensemble Learning experiment
- The “utils” folder contains all utility useful for others files.py. In particular, it contains:
    - skfeature folder contains other function useful to implement Filter techniques
    - config.yml provides paths
    - constants.py provides useful constants to use in different files.py
    - feature_selection.py applies Feature Selection techniques
    - models_and_hyperparameters.py fetches configuration Model/Feature-Selection-Techniques with hyperparameters
    - utilities.py provides utilities used from different files.py
- The file main_demonstrator.py which fetches dataset and starts demonstrator.
- The file main_experiments.py which fetches dataset and starts experiments.

# How to run

1. Run the main_demonstrator.py or main_experiments.py
2. In case of running main_demonstrator.py it automatically shows the results, otherwise main_experiments.py requires to
   respond different questions in order to perform desired experiments