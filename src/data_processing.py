import pandas as pd
import numpy as np
from typing import Tuple, Dict, List 
import logging 
from pathlib import Path 

class DataProcessor:
    """
    A class to handle all the data preprocessing and feature engineering
    of Manitoba soil dataset

    Attributes: 
    data_path (Path): path to raw data file
    target_column (str): Name of the target variable('AGRI_CAP1')
    random_state (int): Random state for reproducibility

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
            data_path (Path): Path to the raw data file
            target_column (str): Name of the target variable
            random_state (int): Random state for reproducibility
        
        """
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.data = None
        self.data_copy = None
        self.X = None
        self.y = None
        

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        try:
            # Create a copy of the original data
            self.data = pd.read_csv(self.data_path)
            self.data_copy = self.data.copy()
            self.logger.info("A copy of the original data has been created.")
        except Exception as e:
            self.logger.error(f'Failed to initialized and copy data: {str(e)}')
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads soil dataset from the specified path 
        Returns: pd.Dataframe: the loaded dataset
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded dataset with the shape {self.data.shape}")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise 

    
    def create_mancon_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates binary flags for each management consideration
        in the MANCON columns.
        """
        management_codes = {
            'F': 'fine_texture',
            'W': 'wetness',
            'T': 'topography',
            'C': 'coarse_texture',
            'B': 'bedrock',
            # Add special cases
            'NO CONSTRAINTS': 'no_constraints',
            'ROCK': 'rock',
            'ORGANIC': 'organic',
            'MARSH' : 'marsh',
            'WATER': 'water'
        }
        
        try:
            # Initialize all flags to 0
            for code_meaning in management_codes.values():
                data[f'has_{code_meaning}'] = 0
            
            # For each MANCON column, check for presence of each code
            for mancon_col in ['MANCON1', 'MANCON2', 'MANCON3']:
                # Clean the data - remove spaces and convert to uppercase
                data[mancon_col] = data[mancon_col].str.strip().str.upper()
                
                for code, meaning in management_codes.items():
                    # Update flag if code is found in the MANCON string
                    mask = data[mancon_col].str.contains(code, na=False, regex=False)
                    data.loc[mask, f'has_{meaning}'] = 1
            
            return data
            
        except Exception as e:
            self.logger.error(f'Error in creating mancon flags: {str(e)}')
            raise
    

    def create_weighted_mancon(self, data: pd.DataFrame) -> pd.Series:
        """
        Function that gives a weighted score to each MANCON column by summing up the 
        total number of management considerations for each row and multiplying it by 
        the EXTENT columns
        eg: FWT has 3 characters(management considerations) -> 3 * EXTENT1 / 100 = W_MANCON1
        """
        management_list = ['F', 'W', 'T', 'C', 'B']
        mancon_columns = ['MANCON1', 'MANCON2', 'MANCON3']
        extent_columns = ['EXTENT1', 'EXTENT2', 'EXTENT3']
        
        try:
            # Fill NaN with empty string to avoid errors in string operations
            data[mancon_columns] = data[mancon_columns].fillna('')

            # Remove whitespace from columns
            data[mancon_columns] = data[mancon_columns].applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
            
            # Process each MANCON-EXTENT pair
            for mancon, extent in zip(mancon_columns, extent_columns):
                # Count total management considerations in each MANCON column
                total_counts = sum(data[mancon].str.count(code) for code in management_list)
                # Calculate weighted score
                data[f'W_{mancon}'] = total_counts * data[extent].fillna(0) / 100
                
            # Create total weighted MANCON column
            data['Total_W_MANCON'] = data['W_MANCON1'] + data['W_MANCON2'] + data['W_MANCON3']
            return data['Total_W_MANCON']
            
        except Exception as e:
            self.logger.error(f'Error in creating Total_weighted_mancon: {str(e)}')
            raise


    def erpoly_processor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        The ERPOLY column is a summary of the ERCLS(water erosion risk class) columns
        This function encodes the ERPOLY column values numerically (N=1, L=2, M=3, H=4, S=5)
        """
        category_mapping = {
        "N": 1,  # Negligible
        "L": 2,  # Low
        "M": 3,  # Moderate
        "H": 4,  # High
        "S": 5   # Severe

        }
        try:
            # Apply the mapping to the column
            # If key isn't found, assign 0 as default value 
            data['encoded_ERPOLY'] = data['ERPOLY'].map(category_mapping).fillna(0)
            
            # Create flag to check if value is missing
            data['ERPOLY_missing'] = data['ERPOLY'].isna().astype(int)

            return data
        
        except Exception as e:
            self.logger.error(f'Error in encoding ERPOLY: {str(e)}')
            raise


    def classification_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method preprocesses the classification columns
        (C_SLOPE, C_DRAIN etc) by subtracting 20 from 
        """
        class_columns = ['C_SLOPE', 'C_DRAIN', 'C_SALT', 'C_SURFTEXT', 'C_AGRI']
        try:
            for c_column in class_columns:
                # Create new columns for encoded values
                encoded_col = f'encoded_{c_column}'
                data[encoded_col] = 0

                # Create mask for standard range (21-28)
                mask = (data[c_column] >= 21) & (data[c_column] <= 28)

                # Encode standard values by subtracting 20
                data.loc[mask, encoded_col] = data.loc[mask, c_column] - 20

                # Create binary flags for special codes 6,7,13,16
                data[f'{c_column}_is_water'] = (data[c_column] == 6).astype(int)
                data[f'{c_column}_is_urban'] = (data[c_column] == 16).astype(int)
                data[f'{c_column}_is_marsh'] = (data[c_column] == 13).astype(int)
                data[f'{c_column}_is_erodedSlope'] = (data[c_column] == 7).astype(int)

                #Log counts for monitoring
                value_counts = data[c_column].value_counts()
                self.logger.info(f"Value counts for {c_column}:\n{value_counts}")

            return data 
        
        except Exception as e:
            self.logger.error(f"Error in encoding class columns: {str(e)}")
            raise 
                
        




            


            