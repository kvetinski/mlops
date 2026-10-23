from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "random-forest-hyperopt"

client = MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT)

if experiment is None:
    raise RuntimeError(f"Experiment not found: {EXPERIMENT}")

# Find the five lowest validation RMSE values
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.rmse >= 0",
    order_by=["metrics.rmse ASC"],
    max_results=5,
)

for rank, run in enumerate(runs, start=1):
    params = {
        name: int(run.data.params[name])
        for name in [
            "max_depth",
            "n_estimators",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
        ]
    }

    print(f"\n#{rank}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Validations RMSE: {run.data.metrics['rmse']:.4f}")
    print(f"Parameters: {params}")
