import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class MarsSeismicMLModel:
    def __init__(self, model_save_path="models/"):
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=15)  # Select top 15 features
        self.label_encoder = LabelEncoder()

        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }

        self.ensemble_model = None
        self.feature_names = None
        self.material_classes = None

        # Create models directory
        os.makedirs(model_save_path, exist_ok=True)

    def prepare_features(self, df):
        """Prepare features for machine learning"""

        # Select relevant features for prediction
        feature_columns = [
            'velocity', 'velocity_squared', 'velocity_log', 'velocity_abs',
            'sampling', 'sampling_log', 'high_frequency',
            'delta', 'delta_inverse', 'delta_log',
            'num_calibrations', 'calibration_density', 'high_calibration',
            'location_numeric', 'station_encoded'
        ]

        # Add time-based features if available
        time_features = ['hour', 'day_of_year', 'month']
        for feature in time_features:
            if feature in df.columns:
                feature_columns.append(feature)

        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]

        if not available_features:
            raise ValueError("No suitable features found for training")

        X = df[available_features].copy()

        # Handle any remaining missing values
        X = X.fillna(X.median())

        # Store feature names
        self.feature_names = available_features

        print(f"Prepared {len(available_features)} features for training")
        return X

    def prepare_target(self, df):
        """Prepare target variable"""
        if 'material_label' in df.columns:
            y = df['material_label'].copy()
        else:
            # Create labels if not available
            y = self.create_material_labels(df)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.material_classes = self.label_encoder.classes_

        print(f"Target variable prepared with {len(self.material_classes)} classes")
        return y_encoded

    def create_material_labels(self, df):
        """Create material labels based on velocity and sampling characteristics"""
        conditions = [
            (df['velocity'] < 0.01) & (df['sampling'] < 50),
            (df['velocity'].between(0.01, 0.05)) & (df['sampling'].between(20, 60)),
            (df['velocity'].between(0.05, 0.15)) & (df['sampling'] > 40),
            (df['velocity'] > 0.15) | (df['sampling'] > 70),
        ]

        choices = [0, 1, 2, 3]  # Sedimentary, Basalt, Dense_Rock, Metallic

        return np.select(conditions, choices, default=4)  # Unknown

    def train_models(self, df):
        """Train all machine learning models"""
        print("=== Training Mars Material Prediction Models ===")

        # Prepare data
        X = self.prepare_features(df)
        y = self.prepare_target(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Train individual models
        trained_models = {}
        model_scores = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train_selected, y_train)

            # Make predictions
            y_pred = model.predict(X_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy

            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)

            print(
                f"{name} - Accuracy: {accuracy:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            trained_models[name] = model

        # Create ensemble model
        self.create_ensemble_model(trained_models, X_train_selected, y_train, X_test_selected, y_test)

        # Save models
        self.save_models(trained_models)

        # Feature importance analysis
        self.analyze_feature_importance(trained_models['random_forest'])

        return model_scores

    def create_ensemble_model(self, trained_models, X_train, y_train, X_test, y_test):
        """Create ensemble model combining all trained models"""
        print("\nCreating ensemble model...")

        # Create voting classifier
        estimators = [(name, model) for name, model in trained_models.items()]

        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability voting
        )

        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)

        # Evaluate ensemble
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

        print(f"Ensemble Model - Accuracy: {ensemble_accuracy:.4f}")

        # Detailed classification report
        print("\nEnsemble Model Classification Report:")
        print(classification_report(y_test, y_pred_ensemble,
                                    target_names=self.material_classes))

    def analyze_feature_importance(self, model):
        """Analyze and save feature importance"""
        if hasattr(model, 'feature_importances_'):
            # Get selected feature names
            selected_features_mask = self.feature_selector.get_support()
            selected_feature_names = [self.feature_names[i] for i, selected in enumerate(selected_features_mask) if
                                      selected]

            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': selected_feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Save feature importance
            importance_df.to_csv(os.path.join(self.model_save_path, 'feature_importance.csv'), index=False)

            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))

    def save_models(self, trained_models):
        """Save all trained models and preprocessing objects"""
        print("\nSaving models...")

        # Save individual models
        for name, model in trained_models.items():
            model_path = os.path.join(self.model_save_path, f'{name}_model.pkl')
            joblib.dump(model, model_path)

        # Save ensemble model
        if self.ensemble_model:
            ensemble_path = os.path.join(self.model_save_path, 'ensemble_model.pkl')
            joblib.dump(self.ensemble_model, ensemble_path)

        # Save preprocessing objects
        scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        selector_path = os.path.join(self.model_save_path, 'feature_selector.pkl')
        joblib.dump(self.feature_selector, selector_path)

        encoder_path = os.path.join(self.model_save_path, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)

        # Save feature names and classes
        metadata = {
            'feature_names': self.feature_names,
            'material_classes': list(self.material_classes),
            'model_info': {
                'n_features': len(self.feature_names),
                'n_classes': len(self.material_classes)
            }
        }

        import json
        metadata_path = os.path.join(self.model_save_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("All models and preprocessing objects saved successfully!")

    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            # Load preprocessing objects
            scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)

            selector_path = os.path.join(self.model_save_path, 'feature_selector.pkl')
            self.feature_selector = joblib.load(selector_path)

            encoder_path = os.path.join(self.model_save_path, 'label_encoder.pkl')
            self.label_encoder = joblib.load(encoder_path)

            # Load ensemble model
            ensemble_path = os.path.join(self.model_save_path, 'ensemble_model.pkl')
            self.ensemble_model = joblib.load(ensemble_path)

            # Load metadata
            metadata_path = os.path.join(self.model_save_path, 'model_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                self.material_classes = metadata['material_classes']

            print("Models loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def predict_material(self, input_data):
        """Predict material type from seismic data"""
        if self.ensemble_model is None:
            raise ValueError("Model not loaded. Please load the model first.")

        # Prepare input data
        if isinstance(input_data, dict):
            # Convert single prediction to DataFrame
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Prepare features
        X = self.prepare_prediction_features(input_df)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Select features
        X_selected = self.feature_selector.transform(X_scaled)

        # Make prediction
        predictions = self.ensemble_model.predict(X_selected)
        probabilities = self.ensemble_model.predict_proba(X_selected)

        # Convert predictions to material names
        predicted_materials = [self.material_classes[pred] for pred in predictions]

        # Create results
        results = []
        for i, (material, probs) in enumerate(zip(predicted_materials, probabilities)):
            prob_dict = {self.material_classes[j]: prob for j, prob in enumerate(probs)}
            results.append({
                'predicted_material': material,
                'confidence': max(probs),
                'probabilities': prob_dict
            })

        return results if len(results) > 1 else results[0]

    def prepare_prediction_features(self, df):
        """Prepare features for prediction (similar to training but without target)"""
        # Create derived features
        self.create_prediction_features(df)

        # Filter for available features
        available_features = [col for col in self.feature_names if col in df.columns]

        if not available_features:
            raise ValueError("No suitable features found for prediction")

        X = df[available_features].copy()

        # Handle missing values
        X = X.fillna(X.median())

        return X

    def create_prediction_features(self, df):
        """Create derived features for prediction"""
        # Velocity-based features
        if 'velocity' in df.columns:
            df['velocity_squared'] = df['velocity'] ** 2
            df['velocity_log'] = np.log1p(np.abs(df['velocity']))
            df['velocity_abs'] = np.abs(df['velocity'])

        # Sampling rate features
        if 'sampling' in df.columns:
            df['sampling_log'] = np.log1p(df['sampling'])
            df['high_frequency'] = (df['sampling'] > 50).astype(int)  # Use fixed threshold

        # Delta features
        if 'delta' in df.columns:
            df['delta_inverse'] = 1 / (df['delta'] + 1e-8)
            df['delta_log'] = np.log1p(df['delta'])

        # Calibration features
        if 'num_calibrations' in df.columns:
            df['calibration_density'] = df['num_calibrations'] / (df['delta'] + 1e-8)
            df['high_calibration'] = (df['num_calibrations'] > 1).astype(int)  # Use fixed threshold

        # Location features
        if 'location' in df.columns:
            df['location_numeric'] = pd.to_numeric(df['location'], errors='coerce').fillna(0)

        # Station features
        if 'station' in df.columns:
            # Simple encoding for new stations
            station_map = {'ELYSE': 0, 'XB': 1}  # Based on your data
            df['station_encoded'] = df['station'].map(station_map).fillna(2)

    def evaluate_model_performance(self, df):
        """Evaluate model performance on test data"""
        if self.ensemble_model is None:
            print("No model loaded for evaluation")
            return None

        # Prepare data
        X = self.prepare_features(df)
        y = self.prepare_target(df)

        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        # Make predictions
        y_pred = self.ensemble_model.predict(X_selected)
        y_pred_proba = self.ensemble_model.predict_proba(X_selected)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)

        # Classification report
        report = classification_report(y, y_pred,
                                       target_names=self.material_classes,
                                       output_dict=True)

        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }


def train_mars_models_from_db(db_path="data/processed/mars_seismic.db"):
    """Convenience function to train models from DuckDB data"""
    import duckdb

    # Connect to database
    conn = duckdb.connect(db_path)

    try:
        # Load data
        df = conn.execute("SELECT * FROM mars_seismic_data").df()

        # Initialize model trainer
        model_trainer = MarsSeismicMLModel()

        # Train models
        scores = model_trainer.train_models(df)

        print(f"\nModel training completed!")
        print("Model performance scores:")
        for model_name, score in scores.items():
            print(f"  {model_name}: {score:.4f}")

        return model_trainer

    except Exception as e:
        print(f"Error training models: {str(e)}")
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    # Train models from processed data
    trainer = train_mars_models_from_db()

    if trainer:
        print("\nModels trained successfully!")
        print("You can now use the Streamlit app for predictions.")
    else:
        print("\nModel training failed. Please run the preprocessing script first.")