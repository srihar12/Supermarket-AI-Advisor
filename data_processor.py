import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class DataProcessor:
    """Handles data preprocessing and validation for supermarket transaction data."""
    
    def __init__(self):
        self.data = None
        self.transaction_col = None
        self.item_col = None
        self.date_col = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data and perform initial validation."""
        try:
            self.data = pd.read_csv(file_path)
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def validate_columns(self, transaction_col: str, item_col: str, date_col: Optional[str] = None) -> bool:
        """Validate that required columns exist in the dataset."""
        required_cols = [transaction_col, item_col]
        if date_col:
            required_cols.append(date_col)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.transaction_col = transaction_col
        self.item_col = item_col
        self.date_col = date_col
        return True
    
    def preprocess_data(self) -> pd.DataFrame:
        """Clean and preprocess the transaction data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Create a copy to avoid modifying original data
        processed_data = self.data.copy()
        
        # Remove rows with missing values in key columns
        processed_data = processed_data.dropna(subset=[self.transaction_col, self.item_col])
        
        # Clean item names (remove extra spaces, standardize case)
        processed_data[self.item_col] = processed_data[self.item_col].astype(str).str.strip().str.title()
        
        # Convert transaction column to string for consistency
        processed_data[self.transaction_col] = processed_data[self.transaction_col].astype(str)
        
        # Process date column if provided
        if self.date_col and self.date_col in processed_data.columns:
            processed_data[self.date_col] = pd.to_datetime(processed_data[self.date_col], errors='coerce')
            processed_data = processed_data.dropna(subset=[self.date_col])
            
            # Add season column
            processed_data['season'] = processed_data[self.date_col].apply(self._get_season)
        
        # Remove duplicate items within the same transaction
        processed_data = processed_data.drop_duplicates(
            subset=[self.transaction_col, self.item_col]
        )
        
        self.data = processed_data
        return processed_data
    
    def _get_season(self, date) -> str:
        """Determine season based on date."""
        if pd.isna(date):
            return 'Unknown'
        
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def get_transaction_baskets(self) -> Dict[str, List[str]]:
        """Convert transaction data to basket format for analysis."""
        if self.data is None:
            raise ValueError("No data processed. Please preprocess data first.")
        
        baskets = {}
        for transaction_id in self.data[self.transaction_col].unique():
            items = self.data[self.data[self.transaction_col] == transaction_id][self.item_col].tolist()
            baskets[transaction_id] = items
        
        return baskets
    
    def get_summary_stats(self) -> Dict:
        """Generate summary statistics for the dataset."""
        if self.data is None:
            raise ValueError("No data available. Please load and process data first.")
        
        stats = {
            'total_transactions': self.data[self.transaction_col].nunique(),
            'total_items': self.data[self.item_col].nunique(),
            'total_rows': len(self.data),
            'avg_items_per_transaction': len(self.data) / self.data[self.transaction_col].nunique()
        }
        
        if self.date_col and self.date_col in self.data.columns:
            stats['date_range'] = {
                'start': self.data[self.date_col].min().strftime('%Y-%m-%d'),
                'end': self.data[self.date_col].max().strftime('%Y-%m-%d')
            }
            
            if 'season' in self.data.columns:
                stats['seasons'] = self.data['season'].value_counts().to_dict()
        
        return stats
    
    def get_top_items(self, n: int = 10) -> pd.DataFrame:
        """Get top N most frequently purchased items."""
        if self.data is None:
            raise ValueError("No data available. Please load and process data first.")
        
        item_counts = self.data[self.item_col].value_counts().head(n)
        return pd.DataFrame({
            'Item': item_counts.index,
            'Count': item_counts.values,
            'Percentage': (item_counts.values / len(self.data) * 100).round(2)
        })
