This project requires two specific folders:

data/ – This folder should contain the avazu data set, available at https://www.kaggle.com/c/avazu-ctr-prediction/data. Note that it it not need to run the python notebook "Experiments synthetic data"
plots/ – This folder will store the generated plots.

The experiments can then be replicated by running the notebooks in the following order.

For the synthetic experiments, running the notebook "Experiments synthetic data" will generate the plots, with no extra steps needed.

For the experiments based on the avazu data set, the notebooks should be run in the following order:
1. "data_process"
2. "0322-deepctr-difm_model_train"

This will generate the reward table.

Then two run the experiments in setting 1 and setting 2 respectively run:
1. "Experiments avazu-one good arm"
2. Experiments avazu-bad arms only"
