
import logging
from pathlib import Path
from src.data_processing import DataProcessor
from src.train_evaluate import XGBoostTrainer # Assuming this is the correct import path

# Configure logging (ensure basicConfig is called only once globally)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__) # Get a logger for this module

def run_pipeline_three_way_split(data_filepath: str, target_column: str):
    """
    Runs the data processing, three-way split (Train+Val / Test),
    hyperparameter optimization (on Train+Val via CV),
    XGBoost training (on Train+Val), and evaluation (on Test) pipeline.

    Args:
        data_filepath (str): The path to the full raw data CSV file (~100k rows).
        target_column (str): The name of the target variable column.
    """
    logger.info("Starting full pipeline execution with three-way split...")

    # --- 1. Data Loading and Processing ---
    logger.info("Initializing DataProcessor and starting data loading and preprocessing...")
    try:
        # Instantiate DataProcessor with the main data path
        data_processor = DataProcessor(
            data_path=Path(data_filepath),
            target_column=target_column
        )

        # Preprocess the data
        processed_data = data_processor.preprocess()
        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed during data loading or processing: {e}")
        return # Stop execution if processing fails


    # --- 2. Three-Way Data Splitting (Train+Val / Test) ---
    logger.info("Splitting data into Train+Validation and Test sets...")
    try:
        # Get the initial split into Train+Validation and Test sets
        # test_size=0.2 means 20% of the *total* data goes to the final test set
        # The remaining 80% is Train+Validation
        X_train_val, X_test, y_train_val, y_test = data_processor.split_data_three_way(test_size=0.2)
        logger.info("Train+Validation / Test split completed.")

    except Exception as e:
        logger.error(f"Pipeline failed during three-way data splitting: {e}")
        return # Stop execution if splitting fails


    # --- 3. Hyperparameter Optimization (on Train+Val data via CV) ---
    logger.info("Initializing XGBoostTrainer for optimization...")
    # Instantiate XGBoostTrainer - Constructor handles MLflow setup
    # Using a clear experiment name for optimization runs
    model_trainer = XGBoostTrainer(experiment_name="soil_viability_optimization_cv")
    logger.info("XGBoostTrainer initialized.")

    logger.info("Starting hyperparameter optimization using cross-validation on Train+Validation data...")
    best_params = None
    try:
        # Use the Train+Validation data (X_train_val, y_train_val) for optimization
        # The optimize_hyperparameters method performs CV on this data internally.
        # Set n_trials and cv_folds as needed.
        optimization_results = model_trainer.optimize_hyperparameters(
            X=X_train_val, # Use the combined Train+Validation set (e.g., 80% of total data)
            y=y_train_val, # Use the combined Train+Validation set
            n_trials=50, # Number of trials for Optuna - Adjust as needed
            cv_folds=5   # Number of CV folds for optimization - Adjust as needed
        )
        best_params = optimization_results['best_params']
        logger.info("Hyperparameter optimization completed on Train+Validation data.")
        logger.info(f"Best parameters found: {best_params}")

    except Exception as e:
        logger.error(f"Pipeline failed during hyperparameter optimization: {e}")
        logger.warning("Proceeding with default/manual parameters for final training.")
        best_params = None # Fallback to default/manual params if optimization fails


    # --- 4. Final Model Training (on Train+Val data with best params) ---
    logger.info("Starting final XGBoost model training on the Train+Validation set...")
    # Use the best parameters found during optimization, if available, otherwise use default/manual
    # Using a clear experiment name for the final model run
    final_model_trainer = XGBoostTrainer(experiment_name="soil_viability_final_model")

    # Define final model parameters - use best_params if available
    final_model_params = best_params if best_params else {
        'objective': 'multi:softmax', # Ensure this matches your problem and DataProcessor target encoding
        'eval_metric': 'mlogloss',    # Appropriate for multi:softmax
        'eta': 0.1,                   # Example default
        'max_depth': 6,               # Example default
        'subsample': 0.8,             # Example default
        'colsample_bytree': 0.8,      # Example default
        'random_state': data_processor.random_state, # Use the random state
        # 'num_class' will be set automatically by XGBoostTrainer.train() in train_evaluate.py
    }

    try:
        # Train the model using the XGBoostTrainer's train method
        # Train on the *entire* Train+Validation set using the best hyperparameters
        trained_model = final_model_trainer.train(X_train_val, y_train_val, params=final_model_params)
        logger.info("Final XGBoost model training completed successfully on Train+Validation set.")
        # The trained_model object is returned by final_model_trainer.train()

    except Exception as e:
        logger.error(f"Pipeline failed during final model training: {e}")
        return # Stop execution if training fails


    # --- 5. Model Evaluation (on Test data) ---
    logger.info("Evaluating the final model on the untouched Test set...")
    try:
        # Evaluate the trained model on the Test set using the XGBoostTrainer's evaluate method
        # This provides the unbiased performance estimate.
        test_metrics = final_model_trainer.evaluate(trained_model, X_test, y_test, dataset_name='final_test')
        logger.info(f"Model evaluation on final test set completed. Metrics: {test_metrics}")

    except Exception as e:
        logger.error(f"Pipeline failed during model evaluation: {e}")
        # Do not return, training might have succeeded.


     #--- 6. Save Model (Optional) ---
        logger.info("Saving the trained model...")
    try:
        # Save the trained model using the XGBoostTrainer's save_model method
        # This method handles MLflow model logging and registration
        # Use the final_model_trainer instance for saving
        model_uri = final_model_trainer.save_model(trained_model, model_name="soil_capability_final_model") # Choose a model name
        logger.info(f"Model saved to MLflow: {model_uri}")
    except Exception as e:
         logger.error(f"Pipeline failed during model saving: {e}")


    logger.info("Full pipeline execution finished.")


if __name__ == "__main__":
    # Define the path to your full dataset and the target column
    data_file = "Soil_Survey_Manitoba.csv"  
    target = "C_AGRI" # Confirm this is the correct target column name

    # Ensure the data file exists
    if not Path(data_file).exists():
        logger.error(f"Data file not found at: {data_file}")
    else:
        run_pipeline_three_way_split(data_file, target)