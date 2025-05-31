import os
import pandas as pd
import numpy as np
import duckdb
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MarsSeismicDataProcessor:
    def __init__(self, raw_data_path="/data/raw/", processed_data_path="/data/processed/"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.db_path = os.path.join(processed_data_path, "mars_seismic.db")

        # Create directories if they don't exist
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)

        # Initialize DuckDB connection
        self.conn = duckdb.connect(self.db_path)

    def extract_data_from_csv(self):
        """Extract data from all CSV files in the raw data directory"""
        csv_files = glob.glob(os.path.join(self.raw_data_path, "*.csv"))

        if not csv_files:
            print("No CSV files found in the raw data directory!")
            return None

        all_data = []

        for file_path in csv_files:
            try:
                print(f"Processing: {os.path.basename(file_path)}")

                # Read CSV file with multiple encoding attempts
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try with error handling
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')

                # Add source file column
                df['source_file'] = os.path.basename(file_path)
                df['processing_timestamp'] = datetime.now()

                all_data.append(df)
                print(f"Successfully loaded {len(df)} records from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined dataset shape: {combined_df.shape}")
            return combined_df
        else:
            print("No data could be extracted from CSV files!")
            return None

    def clean_data(self, df):
        """Clean and preprocess the extracted data"""
        print("Starting data cleaning process...")

        # Make a copy for cleaning
        cleaned_df = df.copy()

        # Standardize column names
        column_mapping = {
            'Network': 'network',
            'Station': 'station',
            'Location': 'location',
            'Channel': 'channel',
            'Start Time': 'start_time',
            'End Time': 'end_time',
            'Sampling': 'sampling',
            'Delta': 'delta',
            'Number of Calibrations': 'num_calibrations',
            'Time': 'calibration_time',
            'Velocity': 'velocity'
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in cleaned_df.columns:
                cleaned_df = cleaned_df.rename(columns={old_name: new_name})

        # Handle missing values
        print("Handling missing values...")

        # Fill missing categorical values
        categorical_cols = ['network', 'station', 'location', 'channel']
        for col in categorical_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('UNKNOWN')

        # Handle numerical columns
        numerical_cols = ['sampling', 'delta', 'num_calibrations', 'velocity']
        for col in numerical_cols:
            if col in cleaned_df.columns:
                # Fill with median for numerical columns
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

        # Handle time columns
        time_cols = ['start_time', 'end_time', 'calibration_time']
        for col in time_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')

        # Create derived features
        print("Creating derived features...")
        self.create_derived_features(cleaned_df)

        # Remove duplicates
        print("Removing duplicates...")
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {initial_rows - len(cleaned_df)} duplicate rows")

        # Remove outliers using IQR method
        print("Removing outliers...")
        cleaned_df = self.remove_outliers(cleaned_df)

        print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
        return cleaned_df

    def create_derived_features(self, df):
        """Create engineered features for better prediction"""

        # Velocity-based features
        if 'velocity' in df.columns:
            df['velocity_squared'] = df['velocity'] ** 2
            df['velocity_log'] = np.log1p(np.abs(df['velocity']))
            df['velocity_abs'] = np.abs(df['velocity'])

            # Velocity categories based on seismic wave types
            df['velocity_category'] = pd.cut(df['velocity'],
                                             bins=[-np.inf, 0.01, 0.05, 0.1, 0.2, np.inf],
                                             labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])

        # Sampling rate features
        if 'sampling' in df.columns:
            df['sampling_log'] = np.log1p(df['sampling'])
            df['high_frequency'] = (df['sampling'] > df['sampling'].median()).astype(int)

        # Delta features
        if 'delta' in df.columns:
            df['delta_inverse'] = 1 / (df['delta'] + 1e-8)  # Avoid division by zero
            df['delta_log'] = np.log1p(df['delta'])

        # Time-based features
        if 'start_time' in df.columns and df['start_time'].notna().any():
            df['hour'] = df['start_time'].dt.hour
            df['day_of_year'] = df['start_time'].dt.dayofyear
            df['month'] = df['start_time'].dt.month

        # Calibration features
        if 'num_calibrations' in df.columns:
            df['calibration_density'] = df['num_calibrations'] / (df['delta'] + 1e-8)
            df['high_calibration'] = (df['num_calibrations'] > df['num_calibrations'].median()).astype(int)

        # Channel-based features
        if 'channel' in df.columns:
            df['channel_type'] = df['channel'].str[:2]  # Extract first 2 characters
            df['channel_direction'] = df['channel'].str[-1]  # Extract last character

        # Location-based features
        if 'location' in df.columns:
            df['location_numeric'] = pd.to_numeric(df['location'], errors='coerce').fillna(0)

        # Station-based features
        if 'station' in df.columns:
            df['station_encoded'] = pd.Categorical(df['station']).codes

        print("Created derived features successfully")

    def remove_outliers(self, df, columns=None):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = ['velocity', 'sampling', 'delta', 'num_calibrations']

        initial_rows = len(df)

        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Keep only rows within bounds
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        print(f"Removed {initial_rows - len(df)} outlier rows")
        return df

    def create_material_labels(self, df):
        """Create material labels based on seismic characteristics"""

        # Define material types based on velocity and other characteristics
        conditions = [
            (df['velocity'] < 0.01) & (df['sampling'] < 50),  # Sedimentary/Loose
            (df['velocity'].between(0.01, 0.05)) & (df['sampling'].between(20, 60)),  # Rock/Basalt
            (df['velocity'].between(0.05, 0.15)) & (df['sampling'] > 40),  # Dense Rock
            (df['velocity'] > 0.15) | (df['sampling'] > 70),  # Metallic/Core
        ]

        choices = ['Sedimentary', 'Basalt', 'Dense_Rock', 'Metallic']

        df['material_type'] = np.select(conditions, choices, default='Unknown')

        # Create numerical labels for ML
        material_mapping = {
            'Sedimentary': 0,
            'Basalt': 1,
            'Dense_Rock': 2,
            'Metallic': 3,
            'Unknown': 4
        }

        df['material_label'] = df['material_type'].map(material_mapping)

        print("Created material labels based on seismic characteristics")
        return df

    def save_to_duckdb(self, df):
        """Save cleaned data to DuckDB"""
        try:
            # Create table and insert data
            self.conn.execute("DROP TABLE IF EXISTS mars_seismic_data")
            self.conn.execute("""
                CREATE TABLE mars_seismic_data AS 
                SELECT * FROM df
            """)

            # Create indexes for better performance
            self.conn.execute("""
                CREATE INDEX idx_velocity ON mars_seismic_data(velocity)
            """)
            self.conn.execute("""
                CREATE INDEX idx_material ON mars_seismic_data(material_type)
            """)
            self.conn.execute("""
                CREATE INDEX idx_station ON mars_seismic_data(station)
            """)

            print(f"Successfully saved {len(df)} records to DuckDB")

            # Save summary statistics
            self.save_data_summary(df)

        except Exception as e:
            print(f"Error saving to DuckDB: {str(e)}")

    def save_data_summary(self, df):
        """Save data summary and statistics"""
        try:
            # Basic statistics
            summary_stats = df.describe()
            summary_stats.to_csv(os.path.join(self.processed_data_path, "data_summary.csv"))

            # Material distribution
            material_dist = df['material_type'].value_counts()
            material_dist.to_csv(os.path.join(self.processed_data_path, "material_distribution.csv"))

            # Station statistics
            station_stats = df.groupby('station').agg({
                'velocity': ['mean', 'std', 'count'],
                'material_type': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
            }).round(4)
            station_stats.to_csv(os.path.join(self.processed_data_path, "station_statistics.csv"))

            print("Saved data summary and statistics")

        except Exception as e:
            print(f"Error saving summary: {str(e)}")

    def get_processed_data(self):
        """Retrieve processed data from DuckDB"""
        try:
            df = self.conn.execute("SELECT * FROM mars_seismic_data").df()
            return df
        except Exception as e:
            print(f"Error retrieving data: {str(e)}")
            return None

    def process_all_data(self):
        """Main method to process all data"""
        print("=== Mars Seismic Data Processing Pipeline ===")

        # Step 1: Extract data
        print("\n1. Extracting data from CSV files...")
        raw_df = self.extract_data_from_csv()

        if raw_df is None:
            print("No data to process. Exiting.")
            return False

        # Step 2: Clean data
        print("\n2. Cleaning and preprocessing data...")
        cleaned_df = self.clean_data(raw_df)

        # Step 3: Create material labels
        print("\n3. Creating material labels...")
        labeled_df = self.create_material_labels(cleaned_df)

        # Step 4: Save to database
        print("\n4. Saving to DuckDB...")
        self.save_to_duckdb(labeled_df)

        # Step 5: Save processed data as CSV backup
        print("\n5. Creating CSV backup...")
        labeled_df.to_csv(os.path.join(self.processed_data_path, "processed_mars_data.csv"), index=False)

        print("\n=== Processing Complete ===")
        print(f"Final dataset shape: {labeled_df.shape}")
        print(f"Material distribution:\n{labeled_df['material_type'].value_counts()}")

        return True

    def close_connection(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Initialize processor
    processor = MarsSeismicDataProcessor()

    # Process all data
    success = processor.process_all_data()

    if success:
        print("\nData processing completed successfully!")
        print("You can now run the Streamlit app: streamlit run app/main.py")
    else:
        print("\nData processing failed. Please check the error messages above.")

    # Close connection
    processor.close_connection()