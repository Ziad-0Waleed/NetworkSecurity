import os
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
from mlflow.models import infer_signature
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow

# import dagshub
# dagshub.init(repo_owner='Ziad-0Waleed', repo_name='NetworkSecurity', mlflow=True)


from urllib.parse import urlparse

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(
        self,
        best_model,
        model_name: str,
        classificationmetric,
        X_eval,
        y_eval,
        stage: str = "test",
        experiment_name: str = "NetworkSecurity",
    ):
        """
        Logs everything needed for a strong MLflow UI experience:
        - Stable experiment name
        - Run name = model name
        - Metrics + params + tags
        - Logged model with signature + input_example
        - Auto evaluation artifacts (plots/metrics) via mlflow.models.evaluate
        """
        try:
            # Put all runs under ONE experiment (project-level)
            mlflow.set_experiment(experiment_name)

            run_name = f"{model_name} ({stage})"

            with mlflow.start_run(run_name=run_name):
                # Tags (useful for filtering/searching in UI)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("stage", stage)

                # Metrics
                mlflow.log_metric("f1_score", classificationmetric.f1_score)
                mlflow.log_metric("precision", classificationmetric.precision_score)
                mlflow.log_metric("recall_score", classificationmetric.recall_score)

                # Params (so Compare view is meaningful)
                # Not all estimators have get_params, but sklearn ones do
                try:
                    mlflow.log_params(best_model.get_params())
                except Exception:
                    pass

                # Model signature + input example (removes the warning)
                input_example = X_eval[:5]
                signature = infer_signature(X_eval, best_model.predict(X_eval))

                model_info = mlflow.sklearn.log_model(
                    best_model,
                    name="model",
                    input_example=input_example,
                    signature=signature,
                )

                # Extra UI visuals (confusion matrix / ROC / PR / metrics) via evaluate
                # Convert X_eval to a DataFrame for evaluation logging
                X_df = pd.DataFrame(X_eval, columns=[f"f{i}" for i in range(X_eval.shape[1])])
                eval_df = X_df.copy()
                eval_df["label"] = y_eval

                mlflow.models.evaluate(
                    model=model_info.model_uri,
                    data=eval_df,
                    targets="label",
                    model_type="classifier",
                )

        except Exception as e:
            # CRITICAL for FastAPI: never let MLflow break /train
            logging.warning(f"MLflow logging failed; continuing without MLflow. Error: {e}")

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=0),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=0),
            "Logistic Regression": LogisticRegression(max_iter=2000, verbose=0),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
            "Random Forest": {"n_estimators": [8, 16, 32, 128, 256]},
            "Gradient Boosting": {
                "learning_rate": [.1, .01, .05, .001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "AdaBoost": {"learning_rate": [.1, .01, .001], "n_estimators": [8, 16, 32, 64, 128, 256]},
        }

        # Put everything under ONE experiment
        mlflow.set_experiment("NetworkSecurity")

        best_model = None
        best_model_name = None
        best_score = -1.0

        # One run per model (easy to sort in UI)
        for model_name, model in models.items():
            grid = GridSearchCV(
                estimator=model,
                param_grid=params.get(model_name, {}),
                scoring="f1",  # selection metric inside grid
                cv=3,
                n_jobs=-1,
                verbose=0
            )

            # Train/tune
            grid.fit(X_train, y_train)
            tuned_model = grid.best_estimator_

            # Predictions
            y_train_pred = tuned_model.predict(X_train)
            y_test_pred = tuned_model.predict(x_test)

            # Metrics (your helper)
            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Log to MLflow as a separate run
            try:
                with mlflow.start_run(run_name=model_name):
                    # hyperparameters
                    mlflow.log_params(grid.best_params_)

                    # metrics (log train + test so you can show overfitting)
                    mlflow.log_metric("f1_train", train_metric.f1_score)
                    mlflow.log_metric("precision_train", train_metric.precision_score)
                    mlflow.log_metric("recall_train", train_metric.recall_score)

                    mlflow.log_metric("f1_test", test_metric.f1_score)
                    mlflow.log_metric("precision_test", test_metric.precision_score)
                    mlflow.log_metric("recall_test", test_metric.recall_score)

                    # model signature + input example (better UI; avoids warning)
                    input_example = x_test[:5]
                    signature = infer_signature(x_test, tuned_model.predict(x_test))

                    mlflow.sklearn.log_model(
                        tuned_model,
                        name="model",
                        input_example=input_example,
                        signature=signature,
                    )

            except Exception as e:
                # Do not break training if MLflow has a problem
                logging.warning(f"MLflow logging failed for {model_name}: {e}")

            # Select best based on test f1 (change rule if you want)
            if test_metric.f1_score > best_score:
                best_score = test_metric.f1_score
                best_model = tuned_model
                best_model_name = model_name
                classification_train_metric = train_metric
                classification_test_metric = test_metric

        logging.info(f"Best model selected: {best_model_name} with test f1={best_score}")

        # Save preprocessor + model
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

        # IMPORTANT: save the INSTANCE, not the class
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )

        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e, sys)