import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SeasonalAnalyzer:
    """Handles seasonal trend analysis for supermarket data."""
    
    def __init__(self):
        self.data = None
        self.seasonal_patterns = None
        self.item_col = None
        self.date_col = None
    
    def prepare_seasonal_data(self, data: pd.DataFrame, item_col: str, date_col: str) -> pd.DataFrame:
        """Prepare data for seasonal analysis."""
        self.data = data.copy()
        self.item_col = item_col
        self.date_col = date_col
        
        # Ensure date column is datetime
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col], errors='coerce')
        
        # Remove rows with invalid dates
        self.data = self.data.dropna(subset=[self.date_col])
        
        # Add time-based features
        self.data['year'] = self.data[self.date_col].dt.year
        self.data['month'] = self.data[self.date_col].dt.month
        self.data['quarter'] = self.data[self.date_col].dt.quarter
        self.data['day_of_week'] = self.data[self.date_col].dt.day_name()
        self.data['season'] = self.data[self.date_col].apply(self._get_season)
        
        return self.data
    
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
    
    def analyze_seasonal_patterns(self) -> Dict:
        """Analyze seasonal patterns for all items."""
        if self.data is None:
            raise ValueError("No seasonal data prepared. Please call prepare_seasonal_data first.")
        
        seasonal_patterns = {}
        
        # Analyze by season
        season_analysis = self.data.groupby(['season', self.item_col]).size().reset_index(name='count')
        
        for item in self.data[self.item_col].unique():
            item_data = season_analysis[season_analysis[self.item_col] == item]
            
            if len(item_data) > 1:  # Only analyze items with data in multiple seasons
                seasonal_counts = item_data.set_index('season')['count'].to_dict()
                
                # Calculate seasonal metrics
                total_count = sum(seasonal_counts.values())
                seasonal_percentages = {
                    season: (count / total_count * 100) if total_count > 0 else 0
                    for season, count in seasonal_counts.items()
                }
                
                # Find peak season
                peak_season = max(seasonal_counts.items(), key=lambda x: x[1])[0]
                
                # Calculate seasonality index (variance from uniform distribution)
                expected_percentage = 25.0  # 25% for each of 4 seasons
                seasonality_index = np.std(list(seasonal_percentages.values()))
                
                seasonal_patterns[item] = {
                    'seasonal_counts': seasonal_counts,
                    'seasonal_percentages': seasonal_percentages,
                    'peak_season': peak_season,
                    'seasonality_index': seasonality_index,
                    'total_purchases': total_count
                }
        
        self.seasonal_patterns = seasonal_patterns
        return seasonal_patterns
    
    def get_seasonal_recommendations(self, season: str, top_n: int = 10) -> List[Dict]:
        """Get recommendations for items to promote in a specific season."""
        if self.seasonal_patterns is None or not self.seasonal_patterns:
            return []
        
        season_items = []
        
        for item, patterns in self.seasonal_patterns.items():
            if season in patterns['seasonal_percentages']:
                season_pct = patterns['seasonal_percentages'][season]
                seasonality_index = patterns['seasonality_index']
                
                # Only include items with significant seasonal variation
                if seasonality_index > 5:  # Threshold for meaningful seasonality
                    season_items.append({
                        'item': item,
                        'season_percentage': season_pct,
                        'seasonality_index': seasonality_index,
                        'peak_season': patterns['peak_season'],
                        'total_purchases': patterns['total_purchases']
                    })
        
        # Sort by seasonal percentage and seasonality index
        season_items.sort(key=lambda x: (x['season_percentage'], x['seasonality_index']), reverse=True)
        
        # Generate recommendations
        recommendations = []
        for item_data in season_items[:top_n]:
            item = item_data['item']
            season_pct = item_data['season_percentage']
            
            if item_data['peak_season'] == season:
                recommendation = f"Promote {item} - it's {season_pct:.1f}% of its annual sales in {season}"
                priority = "High"
            else:
                recommendation = f"Consider {item} - {season_pct:.1f}% of sales occur in {season}"
                priority = "Medium"
            
            recommendations.append({
                'item': item,
                'recommendation': recommendation,
                'priority': priority,
                'season_percentage': season_pct,
                'seasonality_index': item_data['seasonality_index']
            })
        
        return recommendations
    
    def get_monthly_trends(self, item: Optional[str] = None) -> Dict:
        """Get monthly trends for specific item or all items."""
        if self.data is None:
            raise ValueError("No seasonal data prepared. Please call prepare_seasonal_data first.")
        
        if item:
            # Analyze specific item
            item_data = self.data[self.data[self.item_col] == item]
            monthly_counts = item_data.groupby('month').size().to_dict()
            
            return {
                'item': item,
                'monthly_counts': monthly_counts,
                'peak_month': max(monthly_counts.items(), key=lambda x: x[1])[0] if monthly_counts else None
            }
        else:
            # Analyze all items
            monthly_trends = {}
            for item_name in self.data[self.item_col].unique():
                item_data = self.data[self.data[self.item_col] == item_name]
                monthly_counts = item_data.groupby('month').size().to_dict()
                
                if monthly_counts:
                    monthly_trends[item_name] = {
                        'monthly_counts': monthly_counts,
                        'peak_month': max(monthly_counts.items(), key=lambda x: x[1])[0]
                    }
            
            return monthly_trends
    
    def get_seasonal_comparison(self) -> pd.DataFrame:
        """Get a comparison of item performance across seasons."""
        if self.seasonal_patterns is None or not self.seasonal_patterns:
            return pd.DataFrame()
        
        comparison_data = []
        
        for item, patterns in self.seasonal_patterns.items():
            row = {'item': item}
            row.update(patterns['seasonal_percentages'])
            row['peak_season'] = patterns['peak_season']
            row['seasonality_index'] = patterns['seasonality_index']
            comparison_data.append(row)
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Fill missing seasons with 0
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            if season not in df.columns:
                df[season] = 0
        
        # Sort by seasonality_index if it exists, otherwise by peak season
        if 'seasonality_index' in df.columns:
            return df.sort_values('seasonality_index', ascending=False)
        else:
            return df.sort_values('peak_season', ascending=False)
    
    def get_top_seasonal_items(self, season: str, top_n: int = 10) -> pd.DataFrame:
        """Get top items for a specific season."""
        if self.data is None:
            return pd.DataFrame()
        
        season_data = self.data[self.data['season'] == season]
        item_counts = season_data[self.item_col].value_counts().head(top_n)
        
        return pd.DataFrame({
            'Item': item_counts.index,
            'Count': item_counts.values,
            'Percentage': (item_counts.values / len(season_data) * 100).round(2)
        })
    
    def analyze_day_of_week_patterns(self) -> Dict:
        """Analyze purchasing patterns by day of the week."""
        if self.data is None:
            return {}
        
        day_patterns = {}
        
        for item in self.data[self.item_col].unique():
            item_data = self.data[self.data[self.item_col] == item]
            day_counts = item_data['day_of_week'].value_counts().to_dict()
            
            # Find peak day
            peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None
            
            day_patterns[item] = {
                'day_counts': day_counts,
                'peak_day': peak_day,
                'total_purchases': sum(day_counts.values())
            }
        
        return day_patterns
