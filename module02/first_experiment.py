
import mlflow

TRACKING_URI = "http://127.0.0.1:5000"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("zoomcamp-module02")

    with mlflow.start_run(run_name="initial-test"):
        mlflow.log_param("model", "linear_regression")
        mlflow.log_param("features", 515)

        # Historical results from Module 1
        mlflow.log_metric("train_rmse", 7.6493)
        mlflow.log_metric("validation_rmse", 7.8118)

        print("Experiment logged successfully!")


if __name__ == "__main__":
    main()
