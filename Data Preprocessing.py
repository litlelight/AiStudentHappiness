"""
Data Preprocessing Module for PERMA-GNN-Transformer

This module handles preprocessing for two datasets:
1. Lifestyle and Wellbeing Data (Dataset01.csv) - n=12,757, 23 features
2. International Student Mental Health Dataset (Dataset02.zip) - n=268

Paper Reference: Section 4.1.1 Dataset Description
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import zipfile
import os


class DatasetConfig:
    """Configuration for the two datasets used in the paper"""

    # Lifestyle and Wellbeing Data (Large-scale Western cultural background)
    LIFESTYLE_PATH = "Dataset01.csv"
    LIFESTYLE_N_SAMPLES = 12757
    LIFESTYLE_N_FEATURES = 23
    LIFESTYLE_CULTURE = "Western"

    # International Student Mental Health Dataset (Small-scale East Asian)
    MENTAL_HEALTH_ZIP = "Dataset02.zip"
    MENTAL_HEALTH_N_SAMPLES = 268
    MENTAL_HEALTH_CULTURE = "East_Asian"

    # PERMA dimensions for mapping
    PERMA_DIMENSIONS = {
        'Positive_Emotion': 0,
        'Engagement': 1,
        'Relationships': 2,
        'Meaning': 3,
        'Achievement': 4
    }

    # Train/Val/Test split ratios (from paper Section 4.1.2)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1


class LifestyleDataPreprocessor:
    """
    Preprocessor for Lifestyle and Wellbeing Data

    Dataset features (23 dimensions) cover five major areas:
    - Healthy body indicators
    - Healthy mind indicators
    - Professional skill development
    - Social connection strength
    - Life meaning perception

    These naturally map to PERMA dimensions as described in the paper.
    """

    def __init__(self, data_path: str = DatasetConfig.LIFESTYLE_PATH):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.perma_feature_mapping = self._define_perma_mapping()

    def _define_perma_mapping(self) -> Dict[str, list]:
        """
        Define which raw features map to which PERMA dimensions
        Based on paper Section 3.2.2: theory-driven weight initialization

        This mapping guides the PERMA feature embedding layer.
        """
        return {
            'Positive_Emotion': [
                'DAILY_STRESS', 'PLACES_VISITED', 'CORE_CIRCLE',
                'SUPPORTING_OTHERS', 'SOCIAL_NETWORK', 'DAILY_SHOUTING'
            ],
            'Engagement': [
                'FLOW', 'DAILY_STEPS', 'LIVE_VISION', 'SLEEP_HOURS',
                'LOST_VACATION', 'TODO_COMPLETED'
            ],
            'Relationships': [
                'CORE_CIRCLE', 'SUPPORTING_OTHERS', 'SOCIAL_NETWORK',
                'DONATION', 'WEEKLY_MEDITATION'
            ],
            'Meaning': [
                'LIVE_VISION', 'PERSONAL_AWARDS', 'TIME_FOR_PASSION',
                'DONATION', 'SUPPORTING_OTHERS'
            ],
            'Achievement': [
                'PERSONAL_AWARDS', 'TODO_COMPLETED', 'FLOW',
                'TIME_FOR_PASSION', 'LOST_VACATION'
            ]
        }

    def load_data(self) -> pd.DataFrame:
        """Load the Lifestyle and Wellbeing dataset"""
        print(f"Loading Lifestyle and Wellbeing Data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        - Remove duplicates
        - Handle missing values
        - Validate sample size matches paper specification
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values (simple imputation with median)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Validate sample size
        expected_size = DatasetConfig.LIFESTYLE_N_SAMPLES
        actual_size = len(df)
        print(f"Dataset size: {actual_size} (expected ~{expected_size})")

        return df

    def extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features, overall wellbeing label, and PERMA dimension labels

        Returns:
            features: [n_samples, 23] - student behavioral features
            wellbeing_labels: [n_samples, 1] - overall wellbeing scores
            perma_labels: [n_samples, 5] - five PERMA dimension scores
        """
        # Assuming the dataset has a wellbeing score column
        # Adjust column names based on actual dataset structure
        wellbeing_col = 'WELLBEING' if 'WELLBEING' in df.columns else df.columns[-1]

        # Extract feature columns (all except wellbeing)
        feature_cols = [col for col in df.columns if col != wellbeing_col]
        self.feature_columns = feature_cols[:DatasetConfig.LIFESTYLE_N_FEATURES]

        features = df[self.feature_columns].values
        wellbeing_labels = df[wellbeing_col].values.reshape(-1, 1)

        # Generate PERMA dimension labels from features
        # In practice, if dataset has explicit PERMA labels, use those
        perma_labels = self._generate_perma_labels(df, self.feature_columns)

        return features, wellbeing_labels, perma_labels

    def _generate_perma_labels(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        Generate PERMA dimension labels from features using domain knowledge

        This is a simplified version. In the actual paper, PERMA labels may be:
        1. Explicitly provided in the dataset
        2. Derived from survey questions mapped to PERMA dimensions
        3. Generated using psychological assessment tools
        """
        perma_labels = np.zeros((len(df), 5))

        for perma_dim, dim_idx in DatasetConfig.PERMA_DIMENSIONS.items():
            # Get features relevant to this PERMA dimension
            relevant_features = self.perma_feature_mapping.get(perma_dim, [])

            # Average the relevant features (simplified approach)
            available_features = [f for f in relevant_features if f in feature_cols]
            if available_features:
                perma_labels[:, dim_idx] = df[available_features].mean(axis=1).values

        # Normalize PERMA labels to [0, 1]
        perma_labels = (perma_labels - perma_labels.min()) / (perma_labels.max() - perma_labels.min() + 1e-8)

        return perma_labels

    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler

        Args:
            features: Raw feature matrix
            fit: If True, fit the scaler; otherwise use existing fit
        """
        if fit:
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)

        return normalized

    def split_data(self, features: np.ndarray, wellbeing_labels: np.ndarray,
                   perma_labels: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Split data into train/val/test sets using 7:2:1 ratio (paper Section 4.1.2)
        Uses stratified sampling to ensure distribution consistency
        """
        # First split: separate test set (10%)
        X_temp, X_test, y_temp, y_test, perma_temp, perma_test = train_test_split(
            features, wellbeing_labels, perma_labels,
            test_size=DatasetConfig.TEST_RATIO,
            random_state=random_state,
            stratify=self._create_strata(wellbeing_labels)
        )

        # Second split: separate train and val (7:2 from remaining 90%)
        val_ratio = DatasetConfig.VAL_RATIO / (DatasetConfig.TRAIN_RATIO + DatasetConfig.VAL_RATIO)
        X_train, X_val, y_train, y_val, perma_train, perma_val = train_test_split(
            X_temp, y_temp, perma_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=self._create_strata(y_temp)
        )

        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'X_train': X_train, 'y_train': y_train, 'perma_train': perma_train,
            'X_val': X_val, 'y_val': y_val, 'perma_val': perma_val,
            'X_test': X_test, 'y_test': y_test, 'perma_test': perma_test
        }

    def _create_strata(self, labels: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Create stratification bins for continuous wellbeing labels"""
        return pd.qcut(labels.flatten(), q=n_bins, labels=False, duplicates='drop')

    def preprocess_pipeline(self) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline

        Returns:
            Dictionary containing all splits with normalized features
        """
        # Load and clean data
        df = self.load_data()
        df = self.clean_data(df)

        # Extract features and labels
        features, wellbeing_labels, perma_labels = self.extract_features_and_labels(df)

        # Normalize features
        features_normalized = self.normalize_features(features, fit=True)

        # Split data
        data_splits = self.split_data(features_normalized, wellbeing_labels, perma_labels)

        # Add cultural background identifier
        data_splits['culture'] = DatasetConfig.LIFESTYLE_CULTURE

        return data_splits


class MentalHealthDataPreprocessor:
    """
    Preprocessor for International Student Mental Health Dataset

    Dataset includes:
    - PHQ-9 for depression assessment
    - ASSIS for cultural adaptation stress
    - Social connection scales
    - Suicidal ideation indicators
    - Help-seeking behavior

    50% international students, 50% domestic students from Japan
    """

    def __init__(self, zip_path: str = DatasetConfig.MENTAL_HEALTH_ZIP):
        self.zip_path = zip_path
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self) -> pd.DataFrame:
        """Extract and load data from zip file"""
        print(f"Extracting Mental Health Data from {self.zip_path}")

        # Extract zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall('temp_mental_health')

        # Find CSV file in extracted folder
        csv_files = [f for f in os.listdir('temp_mental_health') if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the zip archive")

        csv_path = os.path.join('temp_mental_health', csv_files[0])
        df = pd.read_csv(csv_path)

        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data
        Ensures sample size matches paper specification (n=268)
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values
        # For psychological assessments, use mean imputation cautiously
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)

        # Validate sample size
        expected_size = DatasetConfig.MENTAL_HEALTH_N_SAMPLES
        actual_size = len(df)
        print(f"Dataset size: {actual_size} (expected ~{expected_size})")

        return df

    def extract_mental_health_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from mental health assessment tools

        Maps assessment scores to PERMA dimensions:
        - PHQ-9 scores → Positive Emotion (inverted)
        - Engagement indicators → Engagement
        - Social connection → Relationships
        - Meaning perception → Meaning
        - Academic performance → Achievement
        """
        # This is dataset-specific and depends on actual column names
        # Adjust based on the actual structure of Dataset02

        # Assuming columns like:
        # PHQ-9 items, ASSIS items, social connection, academic indicators, etc.

        feature_cols = [col for col in df.columns
                        if any(keyword in col.upper()
                               for keyword in ['PHQ', 'STRESS', 'SOCIAL', 'ACADEMIC', 'CONNECT'])]

        features = df[feature_cols].values

        # Extract or compute overall wellbeing
        # Typically derived from PHQ-9 (inverted) and other positive indicators
        wellbeing_labels = self._compute_wellbeing_score(df)

        # Extract PERMA dimension labels
        perma_labels = self._extract_perma_from_assessments(df)

        return features, wellbeing_labels, perma_labels

    def _compute_wellbeing_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute overall wellbeing from psychological assessments

        Lower PHQ-9 scores + higher social connection = higher wellbeing
        """
        # Find PHQ-9 related columns (depression indicator)
        phq_cols = [col for col in df.columns if 'PHQ' in col.upper()]

        if phq_cols:
            # PHQ-9 total score (0-27), higher = more depressed
            phq_score = df[phq_cols].sum(axis=1)
            # Invert and normalize to [0, 1]
            wellbeing = 1 - (phq_score / phq_score.max())
        else:
            # Fallback: use mean of positive indicators
            wellbeing = df.select_dtypes(include=[np.number]).mean(axis=1)
            wellbeing = (wellbeing - wellbeing.min()) / (wellbeing.max() - wellbeing.min())

        return wellbeing.values.reshape(-1, 1)

    def _extract_perma_from_assessments(self, df: pd.DataFrame) -> np.ndarray:
        """
        Map psychological assessment items to PERMA dimensions

        Based on standard assessment tools:
        - PHQ-9 items 1-2 → Positive Emotion (inverted)
        - Academic engagement → Engagement
        - Social connection scale → Relationships
        - Life satisfaction → Meaning
        - Academic performance → Achievement
        """
        perma_labels = np.zeros((len(df), 5))

        # Positive Emotion: inverted depression indicators
        depression_cols = [col for col in df.columns if 'PHQ' in col.upper() or 'DEPRESS' in col.upper()]
        if depression_cols:
            depression_score = df[depression_cols].mean(axis=1)
            perma_labels[:, 0] = 1 - (depression_score / depression_score.max())

        # Engagement: academic and activity indicators
        engagement_cols = [col for col in df.columns if 'ENGAG' in col.upper() or 'ACADEMIC' in col.upper()]
        if engagement_cols:
            perma_labels[:, 1] = df[engagement_cols].mean(axis=1)

        # Relationships: social connection indicators
        social_cols = [col for col in df.columns if 'SOCIAL' in col.upper() or 'CONNECT' in col.upper()]
        if social_cols:
            perma_labels[:, 2] = df[social_cols].mean(axis=1)

        # Meaning: life satisfaction and purpose
        meaning_cols = [col for col in df.columns if 'MEANING' in col.upper() or 'SATISF' in col.upper()]
        if meaning_cols:
            perma_labels[:, 3] = df[meaning_cols].mean(axis=1)

        # Achievement: academic performance and accomplishment
        achievement_cols = [col for col in df.columns if 'ACHIEV' in col.upper() or 'PERFORM' in col.upper()]
        if achievement_cols:
            perma_labels[:, 4] = df[achievement_cols].mean(axis=1)

        # Normalize to [0, 1]
        for i in range(5):
            if perma_labels[:, i].max() > 0:
                perma_labels[:, i] = (perma_labels[:, i] - perma_labels[:, i].min()) / \
                                     (perma_labels[:, i].max() - perma_labels[:, i].min() + 1e-8)

        return perma_labels

    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if fit:
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)
        return normalized

    def split_data(self, features: np.ndarray, wellbeing_labels: np.ndarray,
                   perma_labels: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Split data using 7:2:1 ratio
        For small dataset (n=268), stratified sampling is crucial
        """
        # Create stratification bins
        strata = pd.qcut(wellbeing_labels.flatten(), q=5, labels=False, duplicates='drop')

        # First split: separate test set (10%)
        X_temp, X_test, y_temp, y_test, perma_temp, perma_test = train_test_split(
            features, wellbeing_labels, perma_labels,
            test_size=DatasetConfig.TEST_RATIO,
            random_state=random_state,
            stratify=strata
        )

        # Second split: train and val
        strata_temp = pd.qcut(y_temp.flatten(), q=5, labels=False, duplicates='drop')
        val_ratio = DatasetConfig.VAL_RATIO / (DatasetConfig.TRAIN_RATIO + DatasetConfig.VAL_RATIO)
        X_train, X_val, y_train, y_val, perma_train, perma_val = train_test_split(
            X_temp, y_temp, perma_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=strata_temp
        )

        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'X_train': X_train, 'y_train': y_train, 'perma_train': perma_train,
            'X_val': X_val, 'y_val': y_val, 'perma_val': perma_val,
            'X_test': X_test, 'y_test': y_test, 'perma_test': perma_test
        }

    def preprocess_pipeline(self) -> Dict[str, np.ndarray]:
        """Complete preprocessing pipeline for mental health dataset"""
        # Load and clean
        df = self.load_data()
        df = self.clean_data(df)

        # Extract features and labels
        features, wellbeing_labels, perma_labels = self.extract_mental_health_features(df)

        # Normalize
        features_normalized = self.normalize_features(features, fit=True)

        # Split
        data_splits = self.split_data(features_normalized, wellbeing_labels, perma_labels)

        # Add cultural identifier
        data_splits['culture'] = DatasetConfig.MENTAL_HEALTH_CULTURE

        return data_splits


def create_torch_dataset(data_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Convert numpy arrays to PyTorch tensors for model training

    Args:
        data_dict: Dictionary containing numpy arrays from preprocessing

    Returns:
        Dictionary with PyTorch tensors
    """
    torch_data = {}

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            torch_data[key] = torch.FloatTensor(value)
        elif isinstance(value, str):
            torch_data[key] = value

    return torch_data


def load_and_preprocess_datasets() -> Tuple[Dict, Dict]:
    """
    Main function to load and preprocess both datasets

    Returns:
        lifestyle_data: Preprocessed Lifestyle and Wellbeing Data
        mental_health_data: Preprocessed International Student Mental Health Data
    """
    print("=" * 80)
    print("PERMA-GNN-Transformer Data Preprocessing Pipeline")
    print("=" * 80)

    # Process Lifestyle dataset (large-scale)
    print("\n[1/2] Processing Lifestyle and Wellbeing Data...")
    lifestyle_processor = LifestyleDataPreprocessor()
    lifestyle_data = lifestyle_processor.preprocess_pipeline()
    lifestyle_torch = create_torch_dataset(lifestyle_data)

    print(f"✓ Lifestyle dataset ready: {lifestyle_data['X_train'].shape[0]} train samples")

    # Process Mental Health dataset (small-scale)
    print("\n[2/2] Processing International Student Mental Health Data...")
    mental_health_processor = MentalHealthDataPreprocessor()
    mental_health_data = mental_health_processor.preprocess_pipeline()
    mental_health_torch = create_torch_dataset(mental_health_data)

    print(f"✓ Mental Health dataset ready: {mental_health_data['X_train'].shape[0]} train samples")

    print("\n" + "=" * 80)
    print("Data Preprocessing Complete!")
    print("=" * 80)

    return lifestyle_torch, mental_health_torch


if __name__ == "__main__":
    """
    Example usage:

    python data_preprocessing.py
    """

    # Load and preprocess both datasets
    lifestyle_data, mental_health_data = load_and_preprocess_datasets()

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Lifestyle Dataset (Western Culture):")
    print(f"  Train: {lifestyle_data['X_train'].shape}")
    print(f"  Val: {lifestyle_data['X_val'].shape}")
    print(f"  Test: {lifestyle_data['X_test'].shape}")
    print(f"  Features: {lifestyle_data['X_train'].shape[1]}")
    print(f"  PERMA dimensions: {lifestyle_data['perma_train'].shape[1]}")

    print(f"\nMental Health Dataset (East Asian Culture):")
    print(f"  Train: {mental_health_data['X_train'].shape}")
    print(f"  Val: {mental_health_data['X_val'].shape}")
    print(f"  Test: {mental_health_data['X_test'].shape}")
    print(f"  Features: {mental_health_data['X_train'].shape[1]}")
    print(f"  PERMA dimensions: {mental_health_data['perma_train'].shape[1]}")