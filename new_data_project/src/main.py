from src.config import dfs_train, dfs_test, regressor_model_configs, classifier_model_configs, filenames, \
    ablation_config
from src.pipeline import Pipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run regressor or classifier pipeline")
    parser.add_argument(
        "--mode", "-m",
        choices=["regressor", "classifier", "all"],
        default="all",
        help="Choose which pipeline to run: regressor, classifier, or all (default: all)"
    )
    args = parser.parse_args()

    if args.mode in ["regressor", "all"]:
        print("Running regressor pipeline...")
        regressor_pipeline = Pipeline(
            TARGET="time_to_close_days",
            dfs_train=dfs_train,
            dfs_test=dfs_test,
            model_configs=regressor_model_configs,
            filenames=filenames
        )
        regressor_pipeline.run(
            task_type="regression",
            ablation=ablation_config["enabled"],
            ablation_features=ablation_config["bool_features"]
        )

    if args.mode in ["classifier", "all"]:
        print("Running classifier pipeline...")
        classifier_pipeline = Pipeline(
            TARGET="merged",
            dfs_train=dfs_train,
            dfs_test=dfs_test,
            model_configs=classifier_model_configs,
            filenames=filenames
        )
        classifier_pipeline.run(
            task_type="classification",
            ablation=ablation_config["enabled"],
            ablation_features=ablation_config["bool_features"]
        )


if __name__ == "__main__":
    main()
