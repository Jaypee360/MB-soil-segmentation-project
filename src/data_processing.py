# src/data_processing.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # Needed for target encoding if C_AGRI isn't 0-based numerical
from collections import Counter # Import Counter to check class counts

class DataProcessor:
    """
    A class to handle data loading, preprocessing, and splitting
    for the Manitoba soil dataset using a three-way split (Train+Validation / Test).
    """

    def __init__(
            self,
            data_path: Path,
            target_column: str = 'C_AGRI',
            random_state: int = 419
    ):
        """
        Initialize the DataProcessor with the data path and configurations.

        Args:
            data_path (Path): Path to the raw data file (subset of 10k rows).
            target_column (str): Name of the target variable.
            random_state (int): Random state for reproducibility.
        """
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state

        self.raw_data = None # Store the original loaded data
        self.processed_data = None # Store data after preprocessing

        self.label_encoder = LabelEncoder() # Initialize label encoder for the target


        # Set up logging (ensure basicConfig is called only once)
        if not logging.getLogger(__name__).handlers:
             logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
             )
        self.logger = logging.getLogger(__name__)

        try:
            # Load data upon initialization
            self.raw_data = self.load_data(self.data_path)
            self.logger.info("DataProcessor initialized and raw data loaded.")

        except Exception as e:
            self.logger.error(f'Failed to initialize or load data: {str(e)}')
            raise

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """
        Loads the dataset from the specified path.

        Args:
            data_path (Path): Path to the raw data file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            self.logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            self.logger.info(f"Successfully loaded dataset with the shape {data.shape}")
            return data
        except FileNotFoundError:
             self.logger.error(f"Data file not found at: {data_path}")
             raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess(self) -> pd.DataFrame:
        """
        Applies all defined preprocessing steps to the raw data.
        Keeps ORIGINL_RM, the target column, and all newly created columns,
        dropping other original columns not explicitly kept or transformed.
        Stores the result in self.processed_data.

        Returns:
            pd.DataFrame: The fully processed dataframe with selected columns.
        """
        self.logger.info("Starting full data preprocessing pipeline...")

        if self.raw_data is None:
            self.logger.error("No raw data loaded.")
            raise ValueError("No raw data loaded for preprocessing.")

        # Work on a copy of the raw data
        processed_data = self.raw_data.copy()

        # Keep track of columns that exist initially
        original_cols_set = set(processed_data.columns)
        # Keep track of columns we definitely want in the final output
        cols_to_keep_final_set = {'ORIGINL_RM', self.target_column}

        # Ensure ORIGINL_RM and target exist initially before processing
        if 'ORIGINL_RM' not in original_cols_set:
             self.logger.warning("'ORIGINL_RM' column not found in raw data.")
             cols_to_keep_final_set.discard('ORIGINL_RM') # Don't try to keep if missing

        if self.target_column not in original_cols_set:
             self.logger.error(f"Target column '{self.target_column}' not found in raw data.")
             raise ValueError(f"Target column '{self.target_column}' not found in raw data.")


        try:
            # --- Apply specific processing steps and track newly created columns ---

            # Step 1: MANCON Flags
            cols_before = set(processed_data.columns)
            processed_data = self.create_mancon_flags(processed_data)
            cols_after = set(processed_data.columns)
            newly_created_step1 = cols_after - cols_before
            cols_to_keep_final_set.update(newly_created_step1)
            self.logger.debug(f"Step 1 created columns: {newly_created_step1}")


            # Step 2: Weighted MANCON
            cols_before = set(processed_data.columns)
            processed_data = self.create_weighted_mancon(processed_data)
            cols_after = set(processed_data.columns)
            newly_created_step2 = cols_after - cols_before
            cols_to_keep_final_set.update(newly_created_step2)
            self.logger.debug(f"Step 2 created columns: {newly_created_step2}")


            # Step 3: ERPOLY Processing
            cols_before = set(processed_data.columns)
            processed_data = self.erpoly_processor(processed_data)
            cols_after = set(processed_data.columns)
            newly_created_step3 = cols_after - cols_before
            cols_to_keep_final_set.update(newly_created_step3)
            self.logger.debug(f"Step 3 created columns: {newly_created_step3}")


            # Step 4: Classification Processing (Standard encoding and flags)
            cols_before = set(processed_data.columns)
            # Note: classification_processing does NOT process the target column C_AGRI for flags
            # as the target is handled by LabelEncoder later.
            processed_data = self.classification_processing(processed_data)
            cols_after = set(processed_data.columns)
            newly_created_step4 = cols_after - cols_before
            cols_to_keep_final_set.update(newly_created_step4)
            self.logger.debug(f"Step 4 created columns: {newly_created_step4}")

            self.logger.info("Specific preprocessing steps applied.")


            # --- Handle remaining categorical features (One-Hot Encoding) ---
            # Identify categorical columns *after* specific processing but before OHE
            # Exclude columns we already decided to keep (like ORIGINL_RM if it's object)
            # and the target column.
            categorical_cols_for_ohe = [
                col for col in processed_data.columns
                if processed_data[col].dtype == 'object'
                   and col not in cols_to_keep_final_set # Don't OHE columns we already want to keep by name/pattern
                   and col != 'OBJECTID' # Explicitly exclude OBJECTID if it survived and is object
            ]

            if categorical_cols_for_ohe:
                 self.logger.info(f"One-Hot Encoding remaining categorical columns: {categorical_cols_for_ohe}")
                 # Handle potential NaNs in categorical columns before one-hot encoding
                 for col in categorical_cols_for_ohe:
                     # Fill NaN with 'Missing' string
                     processed_data[col] = processed_data[col].fillna('Missing')

                 cols_before_ohe = set(processed_data.columns)
                 # Use dummy_na=False as NaNs are filled with 'Missing' string
                 processed_data = pd.get_dummies(processed_data, columns=categorical_cols_for_ohe, dummy_na=False)
                 cols_after_ohe = set(processed_data.columns)
                 newly_created_by_ohe = cols_after_ohe - cols_before_ohe
                 cols_to_keep_final_set.update(newly_created_by_ohe) # Add OHE columns to the keep list
                 self.logger.debug(f"OHE created columns: {newly_created_by_ohe}")
            else:
                 self.logger.info("No remaining categorical columns to One-Hot Encode.")


            # --- Handle remaining numerical missing values if any ---
            # This applies to any numerical column left (original or newly created)
            numerical_cols_with_nan = processed_data.select_dtypes(include=np.number).columns[processed_data.select_dtypes(include=np.number).isna().any()]
            if numerical_cols_with_nan.size > 0:
                self.logger.warning(f"Numerical columns with missing values after processing: {list(numerical_cols_with_nan)}. Filling with median.")
                for col in numerical_cols_with_nan:
                    if not processed_data[col].isna().all():
                         processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    else:
                         self.logger.warning(f"Column '{col}' is all NaN, cannot fill with median. Consider dropping or other imputation.")


            # --- Target Encoding ---
            # Encode the target column if it's not already numerical and 0-based
            # This step remains the same as it's crucial for the model, regardless of other columns
            # It ensures the target column is in the correct format and handles NaNs in target.
            if self.target_column in processed_data.columns:
                 # Check if target needs encoding (is not numeric or not 0-based)
                 if not pd.api.types.is_numeric_dtype(processed_data[self.target_column]) or (processed_data[self.target_column].min() != 0 if pd.api.types.is_numeric_dtype(processed_data[self.target_column]) else True):
                    self.logger.info(f"Encoding target column '{self.target_column}' using LabelEncoder.")
                    # Ensure target values are strings for LabelEncoder and handle NaNs
                    target_values_str = processed_data[self.target_column].astype(str).replace('nan', np.nan)
                    valid_target_values = target_values_str.dropna()

                    if not valid_target_values.empty:
                        self.label_encoder.fit(valid_target_values)
                        if processed_data[self.target_column].isna().any():
                             self.logger.warning(f"NaN values found in target column '{self.target_column}'. Dropping rows with NaN target.")
                             processed_data.dropna(subset=[self.target_column], inplace=True)
                             # Refit encoder on cleaned data if needed
                             target_values_str = processed_data[self.target_column].astype(str).replace('nan', np.nan)
                             valid_target_values = target_values_str.dropna()
                             if not valid_target_values.empty: self.label_encoder.fit(valid_target_values)

                        if not valid_target_values.empty:
                             # Transform the target column using the fitted encoder
                             processed_data[self.target_column] = self.label_encoder.transform(processed_data[self.target_column].astype(str))
                             self.logger.info(f"Encoded target classes (mapping): {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
                             self.logger.info(f"Encoded target values sample: {processed_data[self.target_column].head()}")
                        else:
                             self.logger.error("No valid target values remaining after handling NaNs. Cannot encode.")
                             raise ValueError("No valid target values after handling NaNs.")
                 else:
                    self.logger.info(f"Target column '{self.target_column}' is already numeric and 0-based or will be handled by XGBoost.")
            else:
                 self.logger.error(f"Target column '{self.target_column}' missing after preprocessing steps.")
                 raise ValueError(f"Target column '{self.target_column}' missing after preprocessing.")


            # --- Final Column Selection ---
            self.logger.info("Performing final column selection...")
            final_cols_set = set(processed_data.columns)
            # Identify columns to drop: all columns currently in the DataFrame
            # MINUS the set of columns we've decided to keep
            cols_to_drop_final = list(final_cols_set - cols_to_keep_final_set)

            if cols_to_drop_final:
                self.logger.info(f"Dropping columns not explicitly kept or newly created ({len(cols_to_drop_final)} columns)...")
                # self.logger.debug(f"Columns being dropped: {cols_to_drop_final}") # Can be very verbose
                processed_data = processed_data.drop(columns=cols_to_drop_final, errors='ignore')
            else:
                self.logger.info("No additional columns to drop based on final selection criteria.")

            self.logger.info(f"Final processed data shape: {processed_data.shape}")


            # Store the processed data
            self.processed_data = processed_data
            self.logger.info("Full data preprocessing pipeline completed.")

            return self.processed_data

        except Exception as e:
            self.logger.error(f"Error during full preprocessing pipeline: {str(e)}")
            raise


    def split_data_three_way(self, test_size: float = 0.2, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the processed data into training+validation and testing sets.
        The training+validation set is used for hyperparameter tuning via CV.
        The test set is reserved for final, unbiased evaluation.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train_val, X_test, y_train_val, y_test
        """
        self.logger.info(f"Splitting processed data into Train+Validation and Test sets (test_size={test_size})...")

        if self.processed_data is None:
             self.logger.error("No processed data available. Call preprocess() first.")
             raise ValueError("No processed data available for splitting.")

        if self.target_column not in self.processed_data.columns:
             self.logger.error(f"Target column '{self.target_column}' not found in processed data.")
             raise ValueError(f"Target column '{self.target_column}' not found.")

        # Separate features (X) and target (y) from the full processed data
        X_full = self.processed_data.drop(columns=[self.target_column])
        y_full = self.processed_data[self.target_column]
        self.logger.info(f"Separated features (shape: {X_full.shape}) and target (shape: {y_full.shape}) for splitting.")


        try:
            # Determine if stratification is possible and appropriate
            stratify_y = None
            if pd.api.types.is_numeric_dtype(y_full) and y_full.nunique() > 1:
                 class_counts = Counter(y_full)
                 min_class_count = min(class_counts.values())

                 self.logger.info(f"Split data: Class counts: {class_counts}") # Debug log
                 self.logger.info(f"Split data: Minimum class count: {min_class_count}") # Debug log

                 # Only stratify if the minimum class size is at least 2
                 # and the number of unique classes is not excessively large
                 if min_class_count >= 2 and y_full.nunique() < 50: # Arbitrary limit for stratification
                     self.logger.info("Stratifying split based on target distribution.")
                     stratify_y = y_full
                 elif min_class_count < 2:
                      self.logger.warning(f"Least populated class in target has only {min_class_count} member(s). Cannot stratify.")
                 elif y_full.nunique() >= 50:
                      self.logger.warning("Target has too many unique values for stratification. Skipping stratification.")
            elif not pd.api.types.is_numeric_dtype(y_full):
                 self.logger.warning("Target is not numeric. Cannot stratify.")
            else:
                 self.logger.warning("Target has only one unique value or is not suitable for stratification. Skipping stratification.")

            self.logger.info(f"Split data: Value of stratify_y before train_test_split: {'None' if stratify_y is None else 'y_full'}") # Debug log


            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_full, y_full,
                test_size=test_size,
                random_state=random_state if random_state is not None else self.random_state,
                stratify=stratify_y # Use stratify_y which is None if stratification is not possible/appropriate
            )

            self.logger.info(f"Split complete. Train+Validation set shape: {X_train_val.shape}, Test set shape: {X_test.shape}")
            # Log distributions only if stratified or few unique values
            if stratify_y is not None or y_full.nunique() < 10:
                self.logger.info(f"Train+Validation target distribution:\n{y_train_val.value_counts(normalize=True)}")
                self.logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


            return X_train_val, X_test, y_train_val, y_test

        except Exception as e:
            self.logger.error(f"Error during three-way data splitting: {str(e)}")
            raise


    # --- Helper processing methods (they accept and return DataFrames) ---

    def create_mancon_flags(self, data: pd.DataFrame) -> pd.DataFrame:
         """
         Creates binary flags for each management consideration.
         (Ensure your implementation takes and returns DataFrame)
         """
         # Example implementation sketch (replace with your actual code)
         management_codes = { 'F': 'fine_texture', 'W': 'wetness', 'T': 'topography', 'C': 'coarse_texture', 'B': 'bedrock',
                              'NO CONSTRAINTS': 'no_constraints', 'ROCK': 'rock', 'ORGANIC': 'organic', 'MARSH' : 'marsh', 'WATER': 'water'}
         for code_meaning in management_codes.values(): data[f'has_{code_meaning}'] = 0
         for mancon_col in ['MANCON1', 'MANCON2', 'MANCON3']:
             if mancon_col in data.columns:
                 # Ensure column is string type before string operations
                 # Replace 'NAN' string with actual NaN before fillna
                 data[mancon_col] = data[mancon_col].astype(str).str.strip().str.upper().replace('NAN', np.nan)
                 for code, meaning in management_codes.items():
                     # Check for presence of code, handling potential NaNs gracefully
                     # Use .str.contains on the string representation, treating NaN as False
                     mask = data[mancon_col].astype(str).str.contains(code, na=False, regex=False)
                     data.loc[mask, f'has_{meaning}'] = 1
         return data

    def create_weighted_mancon(self, data: pd.DataFrame) -> pd.DataFrame:
         """
         Calculates the weighted management consideration score.
         (Ensure your implementation takes and returns DataFrame)
         """
         # Example implementation sketch (replace with your actual code)
         management_list = ['F', 'W', 'T', 'C', 'B']
         mancon_columns = ['MANCON1', 'MANCON2', 'MANCON3']
         extent_columns = ['EXTENT1', 'EXTENT2', 'EXTENT3']
         # Use .map instead of .applymap
         # Fill NaN with empty string for string operations, then strip spaces
         data[mancon_columns] = data[mancon_columns].fillna('').map(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
         for extent_col in extent_columns:
             if extent_col in data.columns: data[extent_col] = pd.to_numeric(data[extent_col], errors='coerce').fillna(0)
             else: data[extent_col] = 0
         for mancon, extent in zip(mancon_columns, extent_columns):
             if mancon in data.columns:
                 # Ensure mancon column is string type for .str accessor
                 # Count occurrences of any code in management_list
                 total_counts = data[mancon].astype(str).str.count('|'.join(management_list)).fillna(0) # Count occurrences of any code
                 data[f'W_{mancon}'] = total_counts * data[extent] / 100
             else: data[f'W_{mancon}'] = 0
         weighted_cols_to_sum = [col for col in [f'W_{mc}' for mc in mancon_columns] if col in data.columns]
         data['Total_W_MANCON'] = data[weighted_cols_to_sum].sum(axis=1)
         return data # Ensure you return the DataFrame


    def erpoly_processor(self, data: pd.DataFrame) -> pd.DataFrame:
         """
         Encodes the ERPOLY column and creates a missing flag.
         (Ensure your implementation takes and returns DataFrame)
         """
         # Example implementation sketch (replace with your actual code)
         category_mapping = {"N": 1, "L": 2, "M": 3, "H": 4, "S": 5}
         if 'ERPOLY' in data.columns:
             data['ERPOLY_missing'] = data['ERPOLY'].isna().astype(int)
             # Map categories to numbers, fill missing (including those not in mapping) with 0
             data['encoded_ERPOLY'] = data['ERPOLY'].map(category_mapping).fillna(0)
         else:
             # If ERPOLY column doesn't exist, create encoded column and missing flag with default values
             data['encoded_ERPOLY'] = 0
             data['ERPOLY_missing'] = 1
         return data # Ensure you return the DataFrame


    def classification_processing(self, data: pd.DataFrame) -> pd.DataFrame:
         """
         Processes classification columns (C_SLOPE etc.)
         by encoding standard values (21-28 -> 1-8) and creating binary flags for special codes (6, 7, 13, 16).
         Accepts a DataFrame and returns the modified DataFrame.
         NOTE: This method does NOT process the target column (C_AGRI) for flags;
         the target is handled by self.label_encoder in the main preprocess method.
         """
         self.logger.info("Processing classification columns...")
         # Define the classification columns to process (excluding the target C_AGRI)
         class_columns = ['C_SLOPE', 'C_DRAIN', 'C_SALT', 'C_SURFTEXT']

         try:
             for c_column in class_columns:
                  if c_column in data.columns: # Check if column exists

                     # Ensure column is numeric, coercing errors to NaN
                     # This is important before numeric comparisons or subtractions
                     original_dtype = data[c_column].dtype
                     # Convert to numeric, coercing errors. Handle potential string representations of numbers.
                     data[c_column] = pd.to_numeric(data[c_column], errors='coerce')
                     if data[c_column].dtype != original_dtype:
                         self.logger.debug(f"Coerced column '{c_column}' to numeric.")


                     # --- Logic for encoding standard range (21-28) ---
                     encoded_col = f'encoded_{c_column}'
                     # Initialize encoded column to NaN first to distinguish from 0 values later
                     if encoded_col not in data.columns:
                          data[encoded_col] = np.nan # Use NaN initially

                     # Create mask for standard range (21-28) and non-NaN
                     # Apply encoding only where the original value is in the standard range
                     mask_standard_range = data[c_column].notna() & (data[c_column] >= 21) & (data[c_column] <= 28)

                     # Encode standard values by subtracting 20 where mask is True
                     data.loc[mask_standard_range, encoded_col] = data.loc[mask_standard_range, c_column] - 20
                     # Fill remaining NaNs in the encoded column (where not in standard range or original NaN) with 0
                     data[encoded_col] = data[encoded_col].fillna(0)
                     # Ensure encoded column is integer type if appropriate
                     # data[encoded_col] = data[encoded_col].astype(int) # Consider if you want int or float


                     # --- Add binary flags for special codes (as requested) ---
                     self.logger.debug(f"Creating special code flags for column: {c_column}")

                     # Define the special codes and their meanings for logging/understanding
                     special_codes = {
                         6: 'water',
                         16: 'urban',
                         13: 'marsh',
                         7: 'erodedSlope'
                     }

                     for code, meaning in special_codes.items():
                         flag_col_name = f'{c_column}_is_{meaning}'
                         # Initialize flag to 0 if it doesn't exist
                         if flag_col_name not in data.columns:
                              data[flag_col_name] = 0

                         # Create a mask for the specific code where the original value is NOT NaN
                         mask_code = data[c_column].notna() & (data[c_column] == code)

                         # Set the flag to 1 where the mask is True
                         # Using |= operator ensures that if the flag was already 1 (e.g. from running twice), it stays 1
                         data[flag_col_name] = data[flag_col_name].astype(bool) | mask_code.astype(bool)
                         data[flag_col_name] = data[flag_col_name].astype(int)

                         # Log counts for specific flags if needed (can be verbose)
                         # if data[flag_col_name].sum() > 0:
                         #      self.logger.debug(f"Count for '{flag_col_name}': {data[flag_col_name].sum()}")


                     # Log counts for monitoring original column values (optional but good for debugging)
                     # value_counts = data[c_column].value_counts(dropna=False).sort_index() # Include NaN counts
                     # self.logger.debug(f"Value counts for original {c_column} after numeric coercion:\n{value_counts}")


                  else:
                      self.logger.warning(f"Classification column '{c_column}' not found in data. Skipping processing for this column.")


             self.logger.info("Finished processing classification columns.")
             return data

         except Exception as e:
             self.logger.error(f"Error in processing classification columns: {str(e)}")
             raise
