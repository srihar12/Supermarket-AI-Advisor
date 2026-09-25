import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import networkx as nx

class Visualizer:
    """Handles all visualizations for the Supermarket AI Advisor."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_association_network(self, rules_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
        """Create an interactive network graph of product associations."""
        if rules_df.empty:
            return go.Figure()
        
        # Get top rules by lift
        top_rules = rules_df.nlargest(top_n, 'lift')
        
        # Create network graph
        fig = go.Figure()
        
        # Add edges (associations)
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            for antecedent in antecedents:
                for consequent in consequents:
                    fig.add_trace(go.Scatter(
                        x=[0, 1],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=rule['lift'] * 2, color=f'rgba(255, {int(255 - rule['lift'] * 50)}, 0, 0.6)'),
                        hoverinfo='none',
                        showlegend=False
                    ))
        
        # Add nodes (products)
        all_items = set()
        for items in top_rules['antecedents']:
            all_items.update(items)
        for items in top_rules['consequents']:
            all_items.update(items)
        
        # Create node positions (simple circular layout)
        n_items = len(all_items)
        angles = np.linspace(0, 2 * np.pi, n_items, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(size=20, color=self.color_palette[:n_items]),
            text=list(all_items),
            textposition='middle center',
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Product Association Network',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_top_sales_chart(self, top_items_df: pd.DataFrame) -> go.Figure:
        """Create a bar chart of top-selling items."""
        if top_items_df.empty:
            return go.Figure()
        
        fig = px.bar(
            top_items_df.head(15),
            x='Count',
            y='Item',
            orientation='h',
            title='Top 15 Best-Selling Products',
            color='Count',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Number of Purchases',
            yaxis_title='Product',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_seasonal_trend_chart(self, seasonal_df: pd.DataFrame, item: Optional[str] = None) -> go.Figure:
        """Create a seasonal trend chart."""
        if seasonal_df.empty:
            return go.Figure()
        
        if item:
            # Single item seasonal trends
            item_data = seasonal_df[seasonal_df['item'] == item]
            if item_data.empty:
                return go.Figure()
            
            fig = go.Figure()
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if 'monthly_counts' in item_data.iloc[0]:
                counts = item_data.iloc[0]['monthly_counts']
                counts_list = [counts.get(month, 0) for month in months]
                
                fig.add_trace(go.Scatter(
                    x=month_names,
                    y=counts_list,
                    mode='lines+markers',
                    name=item,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
        else:
            # Multiple items comparison
            fig = go.Figure()
            
            seasons = ['Winter', 'Spring', 'Summer', 'Fall']
            for season in seasons:
                if season in seasonal_df.columns:
                    fig.add_trace(go.Box(
                        y=seasonal_df[season],
                        name=season,
                        boxpoints='outliers'
                    ))
        
        fig.update_layout(
            title=f'Seasonal Trends{f" for {item}" if item else ""}',
            xaxis_title='Season' if not item else 'Month',
            yaxis_title='Purchase Count',
            height=400
        )
        
        return fig
    
    def create_customer_segment_chart(self, cluster_summary: pd.DataFrame) -> go.Figure:
        """Create a pie chart of customer segments."""
        if cluster_summary.empty:
            return go.Figure()
        
        fig = px.pie(
            cluster_summary,
            values='Customer_Count',
            names='Cluster',
            title='Customer Segment Distribution',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>Cluster %{label}</b><br>Customers: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        return fig
    
    def create_cluster_scatter_plot(self, pca_features: np.ndarray, cluster_labels: np.ndarray) -> go.Figure:
        """Create a scatter plot of customer clusters using PCA."""
        if len(pca_features) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        unique_clusters = np.unique(cluster_labels)
        colors = self.color_palette[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            fig.add_trace(go.Scatter(
                x=pca_features[mask, 0],
                y=pca_features[mask, 1],
                mode='markers',
                marker=dict(size=8, color=colors[i]),
                name=f'Cluster {cluster}',
                hovertemplate=f'Cluster {cluster}<br>PC1: %{{x}}<br>PC2: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Customer Segments (PCA Visualization)',
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            height=500
        )
        
        return fig
    
    def create_lift_confidence_scatter(self, rules_df: pd.DataFrame) -> go.Figure:
        """Create a scatter plot of lift vs confidence for association rules."""
        if rules_df.empty:
            return go.Figure()
        
        fig = px.scatter(
            rules_df,
            x='confidence',
            y='lift',
            size='support',
            hover_data=['antecedents_str', 'consequents_str'],
            title='Association Rules: Lift vs Confidence',
            color='lift',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Confidence',
            yaxis_title='Lift',
            height=500
        )
        
        return fig
    
    def create_seasonal_heatmap(self, seasonal_comparison: pd.DataFrame) -> go.Figure:
        """Create a heatmap of seasonal item performance."""
        if seasonal_comparison.empty:
            return go.Figure()
        
        # Select top 20 items by seasonality index
        top_items = seasonal_comparison.head(20)
        
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        season_data = []
        
        for _, row in top_items.iterrows():
            item_data = [row.get(season, 0) for season in seasons]
            season_data.append(item_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=season_data,
            x=seasons,
            y=top_items['item'].tolist(),
            colorscale='viridis',
            hoverongaps=False,
            hovertemplate='Season: %{x}<br>Item: %{y}<br>Percentage: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Seasonal Item Performance Heatmap (Top 20 Seasonal Items)',
            xaxis_title='Season',
            yaxis_title='Item',
            height=600
        )
        
        return fig
    
    def create_metrics_dashboard(self, metrics: Dict) -> go.Figure:
        """Create a dashboard showing key metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transaction Metrics', 'Item Metrics', 'Analysis Results', 'Performance'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Transaction metrics
        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=metrics.get('total_transactions', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Total Transactions"},
            gauge={'axis': {'range': [None, metrics.get('total_transactions', 100) * 1.2]}}
        ), row=1, col=1)
        
        # Item metrics
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=metrics.get('total_items', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Unique Items"},
            gauge={'axis': {'range': [None, metrics.get('total_items', 50) * 1.2]}}
        ), row=1, col=2)
        
        # Analysis results
        fig.add_trace(go.Indicator(
            mode="number",
            value=metrics.get('association_rules_count', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Association Rules Found"}
        ), row=2, col=1)
        
        # Performance metrics
        avg_lift = metrics.get('avg_lift', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=avg_lift,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Lift"},
            gauge={'axis': {'range': [None, max(5, avg_lift * 1.5)]}}
        ), row=2, col=2)
        
        fig.update_layout(
            height=600,
            title_text="Key Performance Metrics Dashboard"
        )
        
        return fig
    
    def create_item_comparison_chart(self, item_data: Dict[str, Dict]) -> go.Figure:
        """Create a comparison chart for multiple items."""
        if not item_data:
            return go.Figure()
        
        fig = go.Figure()
        
        items = list(item_data.keys())
        metrics = ['total_purchases', 'seasonality_index']
        
        for metric in metrics:
            values = [item_data[item].get(metric, 0) for item in items]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=items,
                y=values
            ))
        
        fig.update_layout(
            title='Item Performance Comparison',
            xaxis_title='Items',
            yaxis_title='Value',
            barmode='group',
            height=400
        )
        
        return fig
