from src.data_preprocessing.data_loading import load_pr_data

dfs_train = [
    load_pr_data("../datas/paddle/github_pr_train_PaddlePaddle_PaddleOCR.csv"),
    load_pr_data("../datas/stockfish/github_pr_train_official-stockfish_Stockfish.csv"),
    load_pr_data("../datas/vscode/microsoft_vscode_pr_dataset_final_split_train.csv"),
    load_pr_data("../datas/pytorch/pytorch_pytorch_pr_dataset_final_split_train.csv")
]
dfs_test = [
    load_pr_data("../datas/paddle/github_pr_test_PaddlePaddle_PaddleOCR.csv"),
    load_pr_data("../datas/stockfish/github_pr_test_official-stockfish_Stockfish.csv"),
    load_pr_data("../datas/vscode/microsoft_vscode_pr_dataset_final_split_train.csv"),
    load_pr_data("../datas/pytorch/pytorch_pytorch_pr_dataset_final_split_test.csv")
]

filenames = ["paddle", "stockfish", 'vscode', 'pytorch']

regressor_model_configs = [
    {"model_type": "LinearRegression"},
    {"model_type": "RandomForest", "n_estimators": 300, "max_depth": 15, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1},
    {"model_type": "GradientBoosting", "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 3, "random_state": 42},
    {"model_type": "HistGradientBoosting", "max_iter": 300, "learning_rate": 0.05, "max_depth": 7, "min_samples_leaf": 3, "early_stopping": True, "random_state": 42},
    {"model_type": "GradientBoosting_log1p", "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 3, "random_state": 42},
]

classifier_model_configs = [
    {"model_type": "LogisticRegression", "max_iter": 1000, "class_weight": "balanced"},
    {"model_type": "RandomForest", "n_estimators": 500, "max_depth": 15, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1},
    {"model_type": "GradientBoosting", "n_estimators": 600, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 3, "random_state": 42},
]

ablation_config = {
    "enabled": False,
    "text_features": ["title"],
    "class_features": ["state"],
    "bool_features": ['has_test', 'has_feature', 'has_bug', 'has_improve', 'has_document', 'has_refactor'],
}


