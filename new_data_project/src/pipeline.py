from src.data_preprocessing.data_loading import load_pr_data
from src.models.model_classifier import MultiProjectClassifier
from src.models.model_regressor import MultiProjectRegressor
from src.data_preprocessing.data_processing import DataPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Pipeline:
    def __init__(self, TARGET, dfs_train, dfs_test, model_configs, filenames):
        self.TARGET = TARGET
        self.model_configs = model_configs
        self.dfs_train = dfs_train
        self.dfs_test = dfs_test
        self.filenames = filenames

    def run(self, task_type="regression", ablation=False, ablation_features=None):
        if not ablation:
            if task_type == "regression":
                self.begin_regressor()
            elif task_type == "classification":
                self.begin_classifier()
        else:
            if task_type == "regression":
                baseline_metrics = self.begin_regressor(ablation_features=None, return_metrics=True)
                ablation_metrics = self.begin_regressor(ablation_features=ablation_features, return_metrics=True)
                all_metrics = baseline_metrics + ablation_metrics
                self.visualize_all_models(all_metrics, task_type="regression", filename_prefix="regression_ablation_compare")

            elif task_type == "classification":
                baseline_metrics = self.begin_classifier(ablation_features=None, return_metrics=True)
                ablation_metrics = self.begin_classifier(ablation_features=ablation_features, return_metrics=True)
                all_metrics = baseline_metrics + ablation_metrics
                self.visualize_all_models(all_metrics, task_type="classification", filename_prefix="classification_ablation_compare")

    def begin_regressor(self, ablation_features=None, return_metrics=False):
        X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
        preprocessor = DataPreprocessor(self.TARGET)

        for idx, (df_train, df_test) in enumerate(zip(self.dfs_train, self.dfs_test)):
            X_train, y_train, num_cols = preprocessor.preprocess_features(df_train, self.filenames[idx])
            X_test, y_test, num_cols = preprocessor.preprocess_features(df_test, self.filenames[idx])

            # 消融去掉特征
            if ablation_features:
                print(f"Removing features {ablation_features} from {self.filenames[idx]}")
                X_train = pd.DataFrame(X_train, columns=num_cols).drop(columns=ablation_features, errors="ignore").values
                X_test = pd.DataFrame(X_test, columns=num_cols).drop(columns=ablation_features, errors="ignore").values

            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)

        all_metrics = []
        for cfg in self.model_configs:
            print(f"\n=== Training model: {cfg['model_type']} ===")
            regressor = MultiProjectRegressor(**cfg)
            regressor.fit(X_train_list, y_train_list, X_test_list, y_test_list)

            for i, m in enumerate(regressor.metrics):
                row = {
                    "Model": cfg['model_type'],
                    "Dataset": f"Dataset {i}",
                    "Ablation": "Removed" if ablation_features else "Baseline",
                    **m
                }
                all_metrics.append(row)

        if return_metrics:
            return all_metrics
        else:
            self.visualize_all_models(all_metrics, task_type="regression",
                                      filename_prefix="all_models_results_ablation" if ablation_features else "all_models_results")

    def begin_classifier(self, ablation_features=None, return_metrics=False):
        X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
        preprocessor = DataPreprocessor(self.TARGET)

        for idx, (df_train, df_test) in enumerate(zip(self.dfs_train, self.dfs_test)):
            X_train, y_train, num_cols = preprocessor.preprocess_features(df_train, self.filenames[idx])
            X_test, y_test, num_cols = preprocessor.preprocess_features(df_test, self.filenames[idx])

            if ablation_features:
                print(f"Removing features {ablation_features} from {self.filenames[idx]}")
                X_train = pd.DataFrame(X_train, columns=num_cols).drop(columns=ablation_features, errors="ignore").values
                X_test = pd.DataFrame(X_test, columns=num_cols).drop(columns=ablation_features, errors="ignore").values

            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)

        all_metrics = []
        for cfg in self.model_configs:
            print(f"\n=== Training model: {cfg['model_type']} ===")
            classifier = MultiProjectClassifier(**cfg)
            classifier.fit(X_train_list, y_train_list, X_test_list, y_test_list)

            for i, m in enumerate(classifier.metrics):
                row = {
                    "Model": cfg['model_type'],
                    "Dataset": f"Dataset {i}",
                    "Ablation": "Removed" if ablation_features else "Baseline",
                    **m
                }
                all_metrics.append(row)

        if return_metrics:
            return all_metrics
        else:
            self.visualize_all_models(all_metrics, task_type="classification",
                                      filename_prefix="all_classifiers_results_ablation" if ablation_features else "all_classifiers_results")

    def visualize_all_models(self, all_metrics, task_type="regression", filename_prefix="all_models"):
        if not all_metrics:
            print("No metrics to visualize")
            return

        df = pd.DataFrame(all_metrics)

        metric_cols = [c for c in df.columns if c not in ["Model", "Dataset", "Ablation"]]
        n_metrics = len(metric_cols)

        output_dir = f"../output/{task_type}"
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        if n_rows * n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in row]

        for i, metric in enumerate(metric_cols):
            ax = axes_flat[i]
            sns.barplot(
                data=df,
                x="Dataset",
                y=metric,
                hue="Ablation",
                ax=ax,
                palette="Blues"
            )
            ax.set_title(f"{metric} by Dataset (Baseline vs Ablation)")
            ax.set_ylabel(metric)
            if task_type == "classification":
                ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{filename_prefix}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Visualization saved to {fig_path}")
