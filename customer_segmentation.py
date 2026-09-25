import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """Handles customer segmentation using K-Means clustering."""
    
    def __init__(self):
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.customer_features = None
        self.cluster_labels = None
        self.optimal_k = None
    
    def prepare_customer_features(self, data: pd.DataFrame, transaction_col: str, item_col: str) -> pd.DataFrame:
        """Create customer-level features for clustering."""
        # Calculate customer-level metrics
        customer_metrics = []
        
        for customer_id in data[transaction_col].unique():
            customer_data = data[data[transaction_col] == customer_id]
            
            metrics = {
                'customer_id': customer_id,
                'total_items_purchased': len(customer_data),
                'unique_items': customer_data[item_col].nunique(),
                'avg_items_per_transaction': len(customer_data) / 1,  # Since we're grouping by transaction
                'transaction_frequency': 1  # Number of transactions for this customer
            }
            
            # Add item category preferences (top items)
            item_counts = customer_data[item_col].value_counts()
            if len(item_counts) > 0:
                top_item = item_counts.index[0]
                metrics[f'top_item_{top_item}'] = item_counts.iloc[0]
            
            customer_metrics.append(metrics)
        
        self.customer_features = pd.DataFrame(customer_metrics)
        
        # Fill NaN values with 0
        self.customer_features = self.customer_features.fillna(0)
        
        return self.customer_features
    
    def find_optimal_clusters(self, max_k: int = 10) -> Dict:
        """Find optimal number of clusters using elbow method."""
        if self.customer_features is None:
            raise ValueError("No customer features prepared. Please call prepare_customer_features first.")
        
        # Select numeric features for clustering
        numeric_features = self.customer_features.select_dtypes(include=[np.number]).columns
        feature_data = self.customer_features[numeric_features]
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Calculate inertia for different k values
        inertias = []
        k_range = range(2, min(max_k + 1, len(feature_data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
        
        # Find optimal k using elbow method (simplified)
        if len(inertias) >= 2:
            # Calculate the rate of change
            diffs = np.diff(inertias)
            rel_diffs = diffs[:-1] / diffs[1:]
            self.optimal_k = k_range[np.argmax(rel_diffs) + 1] if len(rel_diffs) > 0 else 3
        else:
            self.optimal_k = 3
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'optimal_k': self.optimal_k
        }
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> Dict:
        """Perform K-Means clustering on customer features."""
        if self.customer_features is None:
            raise ValueError("No customer features prepared. Please call prepare_customer_features first.")
        
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_clusters()
            n_clusters = self.optimal_k
        
        # Select numeric features for clustering
        numeric_features = self.customer_features.select_dtypes(include=[np.number]).columns
        feature_data = self.customer_features[numeric_features]
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to customer features
        self.customer_features['cluster'] = self.cluster_labels
        
        # Perform PCA for visualization
        self.pca = PCA(n_components=2)
        pca_features = self.pca.fit_transform(scaled_features)
        
        return {
            'n_clusters': n_clusters,
            'cluster_centers': self.kmeans.cluster_centers_,
            'inertia': self.kmeans.inertia_,
            'pca_features': pca_features
        }
    
    def analyze_clusters(self, data: pd.DataFrame, transaction_col: str, item_col: str) -> Dict:
        """Analyze and characterize each customer segment."""
        if self.cluster_labels is None:
            raise ValueError("No clustering performed. Please call perform_clustering first.")
        
        cluster_analysis = {}
        
        for cluster_id in range(len(np.unique(self.cluster_labels))):
            cluster_customers = self.customer_features[
                self.customer_features['cluster'] == cluster_id
            ]['customer_id'].tolist()
            
            # Get all transactions for customers in this cluster
            cluster_transactions = data[data[transaction_col].isin(cluster_customers)]
            
            # Analyze purchasing behavior
            cluster_stats = {
                'cluster_id': cluster_id,
                'customer_count': len(cluster_customers),
                'total_transactions': len(cluster_transactions),
                'avg_items_per_customer': len(cluster_transactions) / len(cluster_customers),
                'unique_items_purchased': cluster_transactions[item_col].nunique(),
                'top_items': cluster_transactions[item_col].value_counts().head(5).to_dict(),
                'avg_transaction_size': cluster_transactions.groupby(transaction_col).size().mean()
            }
            
            # Create segment name based on behavior
            segment_name = self._generate_segment_name(cluster_stats)
            cluster_stats['segment_name'] = segment_name
            
            cluster_analysis[cluster_id] = cluster_stats
        
        return cluster_analysis
    
    def _generate_segment_name(self, cluster_stats: Dict) -> str:
        """Generate a descriptive name for the customer segment."""
        top_items = list(cluster_stats['top_items'].keys())
        avg_items = cluster_stats['avg_items_per_customer']
        
        if len(top_items) == 0:
            return "New Customers"
        
        # Categorize based on top items and behavior
        if any(item.lower() in ['milk', 'bread', 'eggs', 'butter'] for item in top_items[:3]):
            if avg_items > 10:
                return "Family Shoppers"
            else:
                return "Daily Essentials Shoppers"
        elif any(item.lower() in ['chips', 'soda', 'candy', 'snacks'] for item in top_items[:3]):
            return "Snack Enthusiasts"
        elif any(item.lower() in ['beer', 'wine', 'liquor'] for item in top_items[:3]):
            return "Social Shoppers"
        elif avg_items > 15:
            return "Bulk Buyers"
        elif avg_items < 5:
            return "Quick Shoppers"
        else:
            return "Regular Shoppers"
    
    def get_customer_recommendations(self, cluster_id: int) -> List[Dict]:
        """Generate marketing recommendations for a specific customer segment."""
        if cluster_id not in range(len(np.unique(self.cluster_labels))):
            return []
        
        recommendations = []
        
        # Generic recommendations based on cluster characteristics
        cluster_recommendations = {
            0: ["Focus on bundle deals for dairy products", "Promote weekly essentials package"],
            1: ["Create snack combo offers", "Display impulse buys near checkout"],
            2: ["Offer bulk purchase discounts", "Provide loyalty program benefits"],
            3: ["Suggest quick meal solutions", "Highlight ready-to-eat options"]
        }
        
        if cluster_id in cluster_recommendations:
            for rec in cluster_recommendations[cluster_id]:
                recommendations.append({
                    'recommendation': rec,
                    'priority': 'High',
                    'target_segment': f"Cluster {cluster_id}"
                })
        
        return recommendations
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get a summary of all customer clusters."""
        if self.cluster_labels is None:
            return pd.DataFrame()
        
        summary = []
        for cluster_id in range(len(np.unique(self.cluster_labels))):
            cluster_data = self.customer_features[self.customer_features['cluster'] == cluster_id]
            summary.append({
                'Cluster': cluster_id,
                'Customer_Count': len(cluster_data),
                'Percentage': f"{len(cluster_data) / len(self.customer_features) * 100:.1f}%",
                'Avg_Items_Per_Customer': cluster_data['total_items_purchased'].mean(),
                'Avg_Unique_Items': cluster_data['unique_items'].mean()
            })
        
        return pd.DataFrame(summary)
