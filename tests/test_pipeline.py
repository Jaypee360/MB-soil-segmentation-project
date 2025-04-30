import pytest
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock # For mocking
from src.pipeline import run_pipeline_three_way_split # Import the function

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper: Create a dummy data file ---
@pytest.fixture(scope="module")
def dummy_data_file(tmp_path_factory):
    """Creates a temporary dummy CSV file for testing."""
    # Added more diverse data for better testing, including potential NaNs and different target values
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'C_AGRI': [0, 1, 0, 2, 1], # More than one class, min count > 1 for split test
        'MANCON1': ['F', 'W', 'T', 'F', np.nan], # Example for data processing mocks
        'EXTENT1': [10, 20, 30, 40, 50],
        'C_SLOPE': [21, 22, 6, 23, 16] # Example for data processing mocks
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path_factory.mktemp("data") / "pipeline_dummy_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# --- Tests for pipeline.py ---

# Corrected patch targets to include 'src.pipeline.'
@patch('src.pipeline.DataProcessor') # Mock the DataProcessor class
@patch('src.pipeline.XGBoostTrainer') # Mock the XGBoostTrainer class
def test_run_pipeline_success(mock_xgb_trainer_class, mock_data_processor_class, dummy_data_file: Path):
    """Tests successful execution of the pipeline."""

    # Configure the mocked DataProcessor instance and its methods
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance # __init__ returns this mock instance

    # Configure the return value of the preprocess method
    # Ensure dummy processed data has columns expected by subsequent steps (e.g., features for training)
    mock_processed_data = pd.DataFrame({
        'processed_feature_1': [1.1, 2.2, 3.3, 4.4, 5.5],
        'processed_feature_2': [10, 20, 30, 40, 50],
        'C_AGRI': [0, 1, 0, 2, 1] # Include target column
    })
    mock_processor_instance.preprocess.return_value = mock_processed_data

    # Configure the return value of the split_data_three_way method
    # Ensure splits have columns matching processed_data minus target
    mock_X_train_val = pd.DataFrame({
        'processed_feature_1': [1.1, 2.2, 4.4, 5.5],
        'processed_feature_2': [10, 20, 40, 50]
    }, index=[0, 1, 3, 4]) # Example indices
    mock_X_test = pd.DataFrame({
        'processed_feature_1': [3.3],
        'processed_feature_2': [30]
    }, index=[2]) # Example index
    mock_y_train_val = pd.Series([0, 1, 2, 1], index=[0, 1, 3, 4])
    mock_y_test = pd.Series([0], index=[2])

    mock_processor_instance.split_data_three_way.return_value = (
        mock_X_train_val, mock_X_test, mock_y_train_val, mock_y_test
    )

    # Configure the mocked XGBoostTrainer instance and its methods
    mock_trainer_instance = MagicMock()
    mock_xgb_trainer_class.return_value = mock_trainer_instance # __init__ returns this mock instance

    # Configure the return value of the optimize_hyperparameters method
    mock_optimization_results = {'best_params': {'eta': 0.1, 'max_depth': 3}}
    mock_trainer_instance.optimize_hyperparameters.return_value = mock_optimization_results

    # Configure the return value of the train method
    mock_trained_model = MagicMock()
    mock_trainer_instance.train.return_value = mock_trained_model

    # Configure the return value of the evaluate method
    mock_test_metrics = {'final_test_accuracy': 0.99}
    mock_trainer_instance.evaluate.return_value = mock_test_metrics

    # Configure the return value of the save_model method
    mock_model_uri = "runs:/dummy_run_id/models/soil_capability_final_model"
    mock_trainer_instance.save_model.return_value = mock_model_uri


    # Run the pipeline function
    run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # --- Assertions: Check if methods were called in the correct order ---

    # DataProcessor should be initialized
    mock_data_processor_class.assert_called_once_with(data_path=Path(dummy_data_file), target_column='C_AGRI')

    # preprocess should be called
    mock_processor_instance.preprocess.assert_called_once()

    # split_data_three_way should be called after preprocess
    mock_processor_instance.split_data_three_way.assert_called_once_with(test_size=0.2) # Check default test_size

    # XGBoostTrainer should be initialized for optimization
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_optimization_cv")

    # optimize_hyperparameters should be called with train_val data
    mock_trainer_instance.optimize_hyperparameters.assert_called_once_with(
        X=mock_X_train_val,
        y=mock_y_train_val,
        n_trials=50, # Check default
        cv_folds=5   # Check default
    )

    # XGBoostTrainer should be initialized for final model training
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_final_model")
    # To check the second call specifically, you might need to inspect call_args_list
    calls = mock_xgb_trainer_class.call_args_list
    assert len(calls) == 2 # Ensure it was called twice

    # train should be called with train_val data and best params
    expected_final_params = mock_optimization_results['best_params'].copy()
    # The pipeline code also adds default params if best_params is None,
    # and the train method adds num_class.
    # For this test, we mock optimize_hyperparameters to return best_params,
    # so the pipeline should use them.
    # The train method will add num_class internally.
    # We need to check if train was called with the parameters *before* num_class is added by train()
    # However, mocking the class and checking the instance's method call is cleaner.
    # Let's assume the second instance created by the mock class is the one used for training
    # This can be tricky with MagicMock if not careful. A more robust way is to capture the instance.
    # Let's simplify the assertion for now based on the mock_trainer_instance used for optimization,
    # assuming it's reused or the mock behavior applies to all instances.
    # A better approach might be to mock the *instances* explicitly if the class is called multiple times.
    # For now, let's check the train call on the *first* mock_trainer_instance, which might be used for both.
    # If the pipeline truly creates a *new* instance for final training, we'd need to capture that second instance.

    # Based on the pipeline code, a *new* XGBoostTrainer instance is created for final training.
    # We need to get that second instance from the mock_xgb_trainer_class calls.
    # Let's re-configure the mock_xgb_trainer_class to return different mocks for each call.
    mock_optimization_trainer_instance = MagicMock()
    mock_final_trainer_instance = MagicMock()
    mock_xgb_trainer_class.side_effect = [mock_optimization_trainer_instance, mock_final_trainer_instance]

    # Re-run the pipeline with the updated mock setup
    # Need to redefine the test function or use a helper, or adjust the fixture scope if needed.
    # Let's adjust the mocks within the test function itself for clarity.

    # Redo the test function with more precise mocking of instances
    mock_data_processor_class = MagicMock()
    mock_xgb_trainer_class = MagicMock()

    # Configure the mock class to return different instances
    mock_optimization_trainer_instance = MagicMock()
    mock_final_trainer_instance = MagicMock()
    mock_xgb_trainer_class.side_effect = [mock_optimization_trainer_instance, mock_final_trainer_instance]

    # Configure the mock instances' methods as before
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance
    mock_processed_data = pd.DataFrame({
        'processed_feature_1': [1.1, 2.2, 3.3, 4.4, 5.5],
        'processed_feature_2': [10, 20, 30, 40, 50],
        'C_AGRI': [0, 1, 0, 2, 1]
    })
    mock_processor_instance.preprocess.return_value = mock_processed_data
    mock_X_train_val = pd.DataFrame({'processed_feature_1': [1.1, 2.2, 4.4, 5.5], 'processed_feature_2': [10, 20, 40, 50]}, index=[0, 1, 3, 4])
    mock_X_test = pd.DataFrame({'processed_feature_1': [3.3], 'processed_feature_2': [30]}, index=[2])
    mock_y_train_val = pd.Series([0, 1, 2, 1], index=[0, 1, 3, 4])
    mock_y_test = pd.Series([0], index=[2])
    mock_processor_instance.split_data_three_way.return_value = (
        mock_X_train_val, mock_X_test, mock_y_train_val, mock_y_test
    )
    mock_optimization_results = {'best_params': {'eta': 0.1, 'max_depth': 3}}
    mock_optimization_trainer_instance.optimize_hyperparameters.return_value = mock_optimization_results
    mock_trained_model = MagicMock()
    mock_final_trainer_instance.train.return_value = mock_trained_model
    mock_test_metrics = {'final_test_accuracy': 0.99}
    mock_final_trainer_instance.evaluate.return_value = mock_test_metrics
    mock_model_uri = "runs:/dummy_run_id/models/soil_capability_final_model"
    mock_final_trainer_instance.save_model.return_value = mock_model_uri


    # Run the pipeline function again with the refined mocks
    run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # --- Assertions with refined mocks ---

    mock_data_processor_class.assert_called_once_with(data_path=Path(dummy_data_file), target_column='C_AGRI')
    mock_processor_instance.preprocess.assert_called_once()
    mock_processor_instance.split_data_three_way.assert_called_once_with(test_size=0.2)

    # Check calls on the specific mock instances
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_optimization_cv")
    mock_optimization_trainer_instance.optimize_hyperparameters.assert_called_once_with(
        X=mock_X_train_val, y=mock_y_train_val, n_trials=50, cv_folds=5
    )

    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_final_model")
    # Check the train call on the final trainer instance
    expected_final_params = mock_optimization_results['best_params']
    # The train method in XGBoostTrainer adds num_class, but the pipeline passes params *without* it.
    # So, we assert based on what the pipeline passes.
    mock_final_trainer_instance.train.assert_called_once_with(mock_X_train_val, mock_y_train_val, params=expected_final_params)


    mock_final_trainer_instance.evaluate.assert_called_once_with(mock_trained_model, mock_X_test, mock_y_test, dataset_name='final_test')
    mock_final_trainer_instance.save_model.assert_called_once_with(mock_trained_model, model_name="soil_capability_final_model")


# Add tests for error cases (e.g., DataProcessor.preprocess raises error, XGBoostTrainer.train raises error)
# Corrected patch targets to include 'src.pipeline.'
@patch('src.pipeline.DataProcessor')
@patch('src.pipeline.XGBoostTrainer')
def test_run_pipeline_preprocessing_fails(mock_xgb_trainer_class, mock_data_processor_class, dummy_data_file: Path):
    """Tests pipeline handles error during preprocessing."""
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance
    mock_processor_instance.preprocess.side_effect = RuntimeError("Preprocessing failed") # Make preprocess raise an error

    # Ensure the pipeline function exits gracefully (or re-raises the specific error if preferred)
    # Based on the pipeline code, it logs the error and returns, so we don't expect an exception here.
    run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # Check that subsequent steps were NOT called
    mock_processor_instance.preprocess.assert_called_once() # Preprocess should be called
    mock_processor_instance.split_data_three_way.assert_not_called()
    mock_xgb_trainer_class.assert_not_called() # Neither trainer instance should be created


# Add more tests for other failure points (splitting fails, training fails, etc.)
# Example: Test when optimization fails
@patch('src.pipeline.DataProcessor')
@patch('src.pipeline.XGBoostTrainer')
def test_run_pipeline_optimization_fails(mock_xgb_trainer_class, mock_data_processor_class, dummy_data_file: Path):
    """Tests pipeline handles error during optimization."""
    mock_processor_instance = MagicMock()
    mock_data_processor_class.return_value = mock_processor_instance
    # Configure successful preprocessing and splitting
    mock_processed_data = pd.DataFrame({
        'processed_feature_1': [1.1, 2.2, 3.3, 4.4, 5.5],
        'processed_feature_2': [10, 20, 30, 40, 50],
        'C_AGRI': [0, 1, 0, 2, 1]
    })
    mock_processor_instance.preprocess.return_value = mock_processed_data
    mock_X_train_val = pd.DataFrame({'processed_feature_1': [1.1, 2.2, 4.4, 5.5], 'processed_feature_2': [10, 20, 40, 50]}, index=[0, 1, 3, 4])
    mock_X_test = pd.DataFrame({'processed_feature_1': [3.3], 'processed_feature_2': [30]}, index=[2])
    mock_y_train_val = pd.Series([0, 1, 2, 1], index=[0, 1, 3, 4])
    mock_y_test = pd.Series([0], index=[2])
    mock_processor_instance.split_data_three_way.return_value = (
        mock_X_train_val, mock_X_test, mock_y_train_val, mock_y_test
    )

    mock_optimization_trainer_instance = MagicMock()
    mock_final_trainer_instance = MagicMock()
    mock_xgb_trainer_class.side_effect = [mock_optimization_trainer_instance, mock_final_trainer_instance]

    # Make optimization raise an error
    mock_optimization_trainer_instance.optimize_hyperparameters.side_effect = RuntimeError("Optimization failed")

    # Run the pipeline function
    run_pipeline_three_way_split(data_filepath=str(dummy_data_file), target_column='C_AGRI')

    # Check that preprocessing and splitting were called
    mock_processor_instance.preprocess.assert_called_once()
    mock_processor_instance.split_data_three_way.assert_called_once()

    # Check that the optimization trainer was created and optimization was attempted
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_optimization_cv")
    mock_optimization_trainer_instance.optimize_hyperparameters.assert_called_once_with(
        X=mock_X_train_val, y=mock_y_train_val, n_trials=50, cv_folds=5
    )

    # Check that the final trainer was created and train/evaluate/save were called
    # The pipeline logs a warning and proceeds with default params if optimization fails.
    mock_xgb_trainer_class.assert_any_call(experiment_name="soil_viability_final_model")
    # The train method should be called on the final trainer instance
    # Since optimization failed, it should use default parameters.
    # We need to get the default params from the XGBoostTrainer class if possible, or assert based on expected defaults.
    # For simplicity in the test, let's just assert that train was called with *some* params.
    mock_final_trainer_instance.train.assert_called_once()
    mock_final_trainer_instance.evaluate.assert_called_once()
    mock_final_trainer_instance.save_model.assert_called_once()

