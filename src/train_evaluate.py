import mlflow
import mlflow.xgboost
import logging
import optuna 
import json
import numpy as np
import pandas as pd
import xgboost as xgb 
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostTrainer:
    """
    XGBoost model trainer with MLflow tracking for soil viability prediction
    """
    
    def __init__(self, 
                 experiment_name: str = "soil_viability_prediction",
                 tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Initialize trainer with MLflow settings
        
        Parameters:
        -----------
        experiment_name : str
            Name for MLflow experiment
        tracking_uri : str
            URI for MLflow tracking server
        """
        self.logger = logging.getLogger(__name__)
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.xgboost.autolog()  # Enable automatic logging

    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_col: str = 'C_AGRI',
                    test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
        """
        Prepare data for training
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        try:
            self.logger.info("Starting data preparation...")

            # Validate input data
            if target_col not in data.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")
             
            # Create copy of data to avoid modifying original
            df = data.copy()
            # Separate features and target 
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Log feature and target distributions
            self.logger.info(f"Feature shapes: {X.shape}")
            self.logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

            # Create train_test split and use stratify to balance class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if len(y.unique()) < 10 else None
            )
            # Log split sizes
            self.logger.info(f"Training set size: {X_train.shape}")
            self.logger.info(f"Test set size: {X_test.shape}")

            self.logger.info("Data preparation completed successfully")

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise 
        
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        Train XGBoost model with MLflow tracking
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Target values
        params : Optional[Dict]
            XGBoost parameters. If None, use default params
            
        Returns:
        --------
        xgb.XGBClassifier
            Trained model
        """
        try:
            self.logger.info("Starting model training...")

            # Validate input data
            if len(X_train) != len(y_train):
                raise ValueError(
                    f"Feature and target sizes don't match: {len(X_train)} vs {len(y_train)}"
                )
            #Verify our class labels are as expected
            unique_classes = sorted(y_train.unique())
            num_classes = len(unique_classes)
            self.logger.info(f"Found {num_classes} unique soil capability classes: {unique_classes}")

            #Get model parameters and ensure num_class matches our data
            model_params = params if params is not None else self._get_default_params()
            model_params["num_class"] = num_classes
            self.logger.info(f"Training with parameters: {model_params}")

            # Initialize and train model
            with mlflow.start_run(nested=True) as run:
                self.logger.info(f"MLflow run ID:{run.info.run_id}")
                mlflow.log_params(model_params)

                model = xgb.XGBClassifier(**model_params, random_state=42)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train)],
                    early_stopping_rounds=10,
                    verbose=100
                )
                # Log the trained model
                mlflow.xgboost.log_model(model, "soil_capability_classifier")


            return model

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise   



    def evaluate(self,
                model: xgb.XGBClassifier,
                X: pd.DataFrame,
                y: pd.Series,
                dataset_name: str = 'validation') -> Dict[str, float]:
        """
        Evaluate model and log metrics to MLflow
        
        Returns:
        --------
        DIct[str, float]
            Dictionary containing evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating model on {dataset_name} dataset...")
            
            with mlflow.start_run(nested=True):
                # Make predictios
                predictions = model.predict(X)

                # Calculate metrics
                metrics = {
                    f"{dataset_name}_accuracy": accuracy_score(y, predictions),
                    f"{dataset_name}_weighted_f1": f1_score(y, predictions, average='weighted'),
                    f"{dataset_name}_macro_f1": f1_score(y, predictions, average='macro')
                }
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Create and Log confusion matrix
                plt.figure(figsize=(8,6))
                confusion = confusion_matrix(y, predictions)
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix ({dataset_name} set)')
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.savefig(f'confusion_matrix_{dataset_name}.png')
                mlflow.log_artifact(f'confusion_matrix_{dataset_name}.png')
                plt.close()

                # Log feature importance 
                self._log_feature_importance(model, X.columns)

                self.logger.info(
                    f"Evaluation on {dataset_name} set is complete. "
                    f"Accuracy: {metrics[f'{dataset_name}_accuracy']:.4f}, "
                    f"Weighted F1: {metrics[f'{dataset_name}_weighted_f1']:.4f}"
                )

                return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise


            
    def save_model(self,
                  model: xgb.XGBRegressor,
                  model_name: str,
                  artifact_path: str = "models") -> str:
        """
        Save model with MLflow
        
        Returns:
        --------
        str
            Model URI in MLflow
        """
        try:
            self.logger.info(f'Saving model as {model_name}...')

            #Log model to MLflow
            with mlflow.start_run(nested=True) as run:
                model_uri = mlflow.xgboost.log_model(
                    model,
                    artifact_path=artifact_path
                ).model_uri

                # Register model in MLflow Model Registry
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name
                )
                self.logger.info(f'Model registered as {model_name} (version {registered_model.version})')

                #Save additional model metadata
                metadata = {
                    'framework': "xgboost",
                    'type':"classifier",
                    'timestamp': datetime.now().isoformat(),
                    'feature_count': model.n_features_in_,
                    'class_count': model.n_classes_
                }

                # Log metadata as json
                with open('model_metadata.json', "w") as f:
                    json.dump(metadata, f)
                mlflow.log_artifact("model_metadata.json")

                return model_uri 
            
        except Exception as e:
            self.logger.error(f'Error saving model: {str(e)}')
            raise



    def _log_feature_importance(self,
                             model: xgb.XGBRegressor,
                             feature_names: List[str]) -> None:
        """Logs feature importance scores and plots to MLflow
        Paramseters: 
        model: xgb.XGBClassifier
               Trained XGBoost model
        feature_names : List[str]
                List of feature names 
        """
        try:
            # Create and sort feature importance Dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Create bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=feature_importance.head(10),
                x='importance',
                y='feature'
            )
            plt.title('Top ten important features')
            plt.tight_layout()

            # Save and log to mlflow
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error in feature importance loggin: {str(e)}")
            raise

        

    def cross_validate(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      params: Optional[Dict] = None,
                      n_splits: int = 5,
                      stratify: bool = True) -> Dict[str, float]:
        """
        Perform k-fold cross-validation and log results

        n_splits: Number of folds in cross-validation
        stratify: bool, whether to use stratified sampling 
        to create balanced folds

        Returns:
        --------
        Dict[str, float]
            Dictionary containing Cross-validation metrics
        """
        try:
            self.logger.info(f"starting {n_splits}-fold cross-validation...")

            # Get model parameters 
            model_params = params if params is not None else self._get_default_params()
            model_params['num_class'] = len(y.unique())

            # Use stratified k-fold for classification
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Create base classifier 
            model = xgb.XGBClassifier(**model_params)

            # Define metrics to evaluate 
            scoring = {
                'accuracy':'accuracy',
                'f1_weighted':'f1_weigted',
                'f1_macro':'f1_macro'
            }

            # Perform cross-validation with miltiple metrics
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False
            )

            # Calculate mean scores
            mean_metrics = {
                'mean_cv_accuracy':np.mean(cv_results['test_accuracy']),
                'mean_cv_f1_weighted':np.mean(cv_results['test_f1_weighted']),
                'mean_cv_f1_macro':np.meean(cv_results['test_f1_macro'])
            }

            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params({'cv_folds': n_splits, **model_params})
                mlflow.log_metrics(mean_metrics)

            self.logger.info(
                f"Cross-validation completed. Mean accuracy: {mean_metrics['mean_cv_accuracy']:.4f}"
            )
            return mean_metrics
        
        except Exception as e:
            self.logger.error(f'Error during cross-validation: {str(e)}')
            raise

    
    def optimize_hyperparameters(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 n_trials: int = 50,
                                 cv_folds: int = 5,
                                 timeout: Optional[int] = None) -> Dict:
        """
        Optimizes hyperparameters using Optuna with cross-validation.

        Params:
        --------
        n_trials: int, default=50
          Number of Optuna trials
        cv_folds: int, default=5
          Number of cross-validation folds
        timeout: Optiona;[int], default=None
          Timeout in seconds. None means no timeout.

        Returns
        ---------
        Dict 
          Dictionary containing best parameters
        """
        try:
            self.logger.info(f'Starting hyperparameter optimization with {n_trials} trials...')

            # Define the objective function for Optuna 
            def objective(trial):
                # Define the hyperparameter search space
                param = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'num_class': len(y.unique()),
                    'objective': 'multi:softmax',
                    'random_state': 42

                }

                # Create the cross-validation strategy 
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                # Intialize scores list
                scores = []

                # Perform cross-validation 
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Train model with current parameters
                    model = xgb.XGBClassifier(**param)
                    model.fit(X_train, y_train)

                    # Evaluate 
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average='weighted')
                    scores.append(score)

                # Return mean score across folds
                return np.mean(scores)
            
            # Create a study to maximize F1 score
            study = optuna.create_study(direction='maximize')

            # Start optimization with MLflow tracking
            with mlflow.start_run(nested=True) as run:
                mlflow.log_params({
                    'optimization_type':'optuna',
                    'n_trials':n_trials,
                    'cv_folds':cv_folds,
                    'timeout':timeout
                })

                # Optimize 
                study.optimize(objective, n_trials=n_trials, timeout=timeout)

                # Get best parameters
                best_params = study.best_params
                best_value = study.best_value

                # Log best parameters and score
                mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
                mlflow.log_metric('best_weighted_f1', best_value)

                # Log optimization history as CSV 
                history_df = study.trials_dataframe()
                history_df.to_csv('optuna_history.csv', index=False)
                mlflow.log_artifact('optuna_history.csv')

                self.logger.info(f'Hyperparameter optimization completed.')
                self.logger.info(f'Best weighted F1 score: {best_value:.4f}')
                self.logger.info(f'Best parameters: {best_params}')

                return {
                    'best_params': best_params,
                    'best_score': best_value
                } 
        except Exception as e:
            self.logger.error(f'Error during hyperparameter optimization: {str(e)}')
            raise
            
    
    
    def _get_default_params(self) -> Dict:
        """
        Returns a dictionary of default parameters for XGBoost classifier
        These parameters provide a reasonable starting point for soil
        capability classification 

        Returns
        -------
        Dict 
        Dictionary containing default XGBoost parameters
        """
        default_params = {
            # Core parameters 
            'objective':'multi:softmax',
            'learning_rate':0.1,
            'max_depth':6,
            'n_estimators':100,

            # Regularization parameters 
            'subsample':0.8,
            'colsample_bytree':0.8,
            'min_child_weight':1,
            'gamma':0,
            'reg_alpha':0,
            'reg_lambda':1,

            # Performance parameters
            'n_jobs':-1,
            'random_state':42,
        }
        self.logger.info("Using default XGBoost classification parameters")
        return default_params
    
if __name__ == "__main__":
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Defining the path to the preprocessed data file and target column
    # The file should be the output of data_processing.py script
    preprocessed_data_path = "10k_preprocessed_soil_data.csv"
    target_variable = "C_AGRI"

    if not Path(preprocessed_data_path).exists():
        logger.error(f"Preprocessed data file not found at: {preprocessed_data_path}")
    else:
        logger.info(f"Starting standalone model training and evaluation with data from: {preprocessed_data_path}")
        try:
            # Load the preprocessed data
            data = pd.read_csv(preprocessed_data_path)
            logger.info(f"Loaded preprocessed data with shape: {data.shape}")

            # Initialize the XGBoostTrainer
            trainer = XGBoostTrainer(experiment_name="standalone_training_run")

            # Prepare the data (split into train and test sets)
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                data=data,
                target_col=target_variable,
                test_size=0.2,
                random_state=419
            )
            logger.info("Data prepared for training")

            # Train the model using default parameters
            # You could also define parameters in a dictionary
            # e.g custom_params = {'learning_rate':0.05, max_depth: 5}
            # trained_model = trainer.train(X_train, y_train, params=custom_params)
            trained_model = trainer.train(X_train, y_train)
            logger.info("Model training complete.")

            # Evaluate the model on the test set
            test_metrics = trainer.evaluate(trained_model, X_test, y_test, dataset_name='test_set')
            logger.info(f"Model evaluation complete. Test Metrics: {test_metrics}")

            # Save the trained model
            model_uri = trainer.save_model(trained_model, model_name='standalone_soil_classifier')
            logger.info(f"Model saved to MLflow: {model_uri}")

        except Exception as e:
            logger.error(f"An error occurred during standalone training/evaluation: {e}")
    


