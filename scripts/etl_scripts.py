import pandas as pd
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_data(filepath):
    """Extract data from a source CSV file."""
    try:
        data = pd.read_csv(filepath)
        logging.info("Data extraction successful.")
        return data
    except Exception as e:
        logging.error(f"Failed to extract data: {e}")
        return None

def transform_data(data):
    """Transform data by converting text to lowercase."""
    try:
        transformed_data = data['text_column'].apply(lambda x: x.lower())
        logging.info("Data transformation successful.")
        return transformed_data
    except Exception as e:
        logging.error(f"Failed to transform data: {e}")
        return None

def load_data(transformed_data, output_filepath):
    """Load transformed data into a destination CSV file."""
    try:
        transformed_data.to_csv(output_filepath, index=False)
        logging.info("Data loading successful.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
