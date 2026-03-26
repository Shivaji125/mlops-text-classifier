import mlflow

mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()

exp = mlflow.get_experiment_by_name("sentiment-classifier")
runs = mlflow.search_runs([exp.experiment_id], order_by=["metrics.test_f1 DESC"], max_results=1)
best_run_id = runs.iloc[0]["run_id"]
print(f"Best run: {best_run_id}")

mv = mlflow.register_model(f"runs:/{best_run_id}/model", "sentiment-classifier")
print(f"Registered version: {mv.version}")

client.transition_model_version_stage("sentiment-classifier", mv.version, "Production")
print(f"Model v{mv.version} promoted to Production")