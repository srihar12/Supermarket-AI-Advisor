import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

class AIChatbot:
    """Interactive AI chatbot for supermarket business intelligence."""
    
    def __init__(self):
        self.data_processor = None
        self.market_basket_analyzer = None
        self.customer_segmentation = None
        self.seasonal_analyzer = None
        self.conversation_history = []
        
    def set_analyzers(self, data_processor, market_basket_analyzer, 
                     customer_segmentation, seasonal_analyzer):
        """Set the analyzer instances for the chatbot to use."""
        self.data_processor = data_processor
        self.market_basket_analyzer = market_basket_analyzer
        self.customer_segmentation = customer_segmentation
        self.seasonal_analyzer = seasonal_analyzer
    
    def process_query(self, query: str) -> Dict:
        """Process user query and generate response."""
        query_lower = query.lower()
        
        # Store query in conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'response': None
        })
        
        # Determine query type and generate response
        if self._is_product_combo_query(query_lower):
            response = self._handle_product_combo_query(query)
        elif self._is_customer_segment_query(query_lower):
            response = self._handle_customer_segment_query(query)
        elif self._is_seasonal_query(query_lower):
            response = self._handle_seasonal_query(query)
        elif self._is_top_products_query(query_lower):
            response = self._handle_top_products_query(query)
        elif self._is_item_association_query(query_lower):
            response = self._handle_item_association_query(query)
        elif self._is_recommendation_query(query_lower):
            response = self._handle_recommendation_query(query)
        elif self._is_metrics_query(query_lower):
            response = self._handle_metrics_query(query)
        else:
            response = self._handle_general_query(query)
        
        # Update conversation history
        self.conversation_history[-1]['response'] = response
        
        return response
    
    def _is_product_combo_query(self, query: str) -> bool:
        """Check if query is about product combinations."""
        keywords = ['combo', 'combination', 'together', 'bought together', 'association']
        return any(keyword in query for keyword in keywords)
    
    def _is_customer_segment_query(self, query: str) -> bool:
        """Check if query is about customer segments."""
        keywords = ['customer', 'segment', 'shopper', 'type of customer']
        return any(keyword in query for keyword in keywords)
    
    def _is_seasonal_query(self, query: str) -> bool:
        """Check if query is about seasonal trends."""
        keywords = ['season', 'seasonal', 'summer', 'winter', 'spring', 'fall', 'promote']
        return any(keyword in query for keyword in keywords)
    
    def _is_top_products_query(self, query: str) -> bool:
        """Check if query is about top products."""
        keywords = ['top', 'best', 'popular', 'selling', 'most purchased']
        return any(keyword in query for keyword in keywords)
    
    def _is_item_association_query(self, query: str) -> bool:
        """Check if query is about specific item associations."""
        keywords = ['what goes with', 'what should i put with', 'association with']
        return any(keyword in query for keyword in keywords)
    
    def _is_recommendation_query(self, query: str) -> bool:
        """Check if query is asking for recommendations."""
        keywords = ['recommend', 'suggest', 'should i', 'what should']
        return any(keyword in query for keyword in keywords)
    
    def _is_metrics_query(self, query: str) -> bool:
        """Check if query is about metrics or statistics."""
        keywords = ['how many', 'what is', 'metrics', 'statistics', 'performance']
        return any(keyword in query for keyword in keywords)
    
    def _ensure_association_rules(self, min_support: float = 0.01, min_confidence: float = 0.5) -> None:
        """Attempt to generate association rules if they are not already available."""
        if self.market_basket_analyzer.rules is not None and not self.market_basket_analyzer.rules.empty:
            return
        if self.data_processor is None or self.data_processor.data is None:
            return
        try:
            baskets = self.data_processor.get_transaction_baskets()
            if baskets:
                self.market_basket_analyzer.prepare_transactions(baskets)
                self.market_basket_analyzer.find_frequent_itemsets(min_support)
                self.market_basket_analyzer.generate_association_rules(min_confidence)
        except Exception:
            pass
    
    def _handle_product_combo_query(self, query: str) -> Dict:
        """Handle queries about product combinations."""
        self._ensure_association_rules()
        if self.market_basket_analyzer.rules is None or self.market_basket_analyzer.rules.empty:
            return {
                'type': 'error',
                'message': 'No association rules available. Please run market basket analysis first.',
                'suggestion': 'Try uploading data and running the analysis first.'
            }
        
        top_rules = self.market_basket_analyzer.get_top_rules(5, 'lift')
        
        if top_rules.empty:
            return {
                'type': 'info',
                'message': 'No significant product combinations found.',
                'suggestion': 'Try adjusting the sensitivity parameters (support/confidence).'
            }
        
        combos = []
        for _, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            
            combos.append({
                'combo': f"{antecedents} → {consequents}",
                'lift': rule['lift'],
                'confidence': f"{rule['confidence']:.1%}",
                'explanation': f"When customers buy {antecedents}, they are {rule['lift']:.1f}x more likely to buy {consequents}."
            })
        
        return {
            'type': 'product_combos',
            'title': 'Best Product Combinations',
            'combos': combos,
            'total_found': len(self.market_basket_analyzer.rules)
        }
    
    def _handle_customer_segment_query(self, query: str) -> Dict:
        """Handle queries about customer segments."""
        if self.customer_segmentation.cluster_labels is None:
            return {
                'type': 'error',
                'message': 'No customer segmentation available. Please run customer analysis first.',
                'suggestion': 'Try running the customer segmentation analysis.'
            }
        
        cluster_summary = self.customer_segmentation.get_cluster_summary()
        
        if cluster_summary.empty:
            return {
                'type': 'info',
                'message': 'No customer segments identified.',
                'suggestion': 'Check if you have sufficient transaction data.'
            }
        
        segments = []
        for _, row in cluster_summary.iterrows():
            segments.append({
                'segment_id': int(row['Cluster']),
                'customer_count': int(row['Customer_Count']),
                'percentage': row['Percentage'],
                'avg_items': f"{row['Avg_Items_Per_Customer']:.1f}",
                'characteristics': f"Customers in this segment buy around {row['Avg_Items_Per_Customer']:.1f} items on average."
            })
        
        return {
            'type': 'customer_segments',
            'title': 'Key Customer Segments',
            'segments': segments,
            'total_customers': len(self.customer_segmentation.customer_features)
        }
    
    def _handle_seasonal_query(self, query: str) -> Dict:
        """Handle queries about seasonal trends."""
        if self.seasonal_analyzer.seasonal_patterns is None:
            return {
                'type': 'error',
                'message': 'No seasonal analysis available. Please run seasonal analysis first.',
                'suggestion': 'Make sure your data includes date information.'
            }
        
        # Determine which season the user is asking about
        season_keywords = {
            'summer': 'Summer',
            'winter': 'Winter',
            'spring': 'Spring',
            'fall': 'Fall',
            'autumn': 'Fall'
        }
        
        target_season = None
        for keyword, season in season_keywords.items():
            if keyword in query:
                target_season = season
                break
        
        if target_season:
            recommendations = self.seasonal_analyzer.get_seasonal_recommendations(target_season, 5)
            return {
                'type': 'seasonal_recommendations',
                'title': f'Recommendations for {target_season}',
                'season': target_season,
                'recommendations': recommendations
            }
        else:
            # General seasonal overview
            seasonal_comparison = self.seasonal_analyzer.get_seasonal_comparison()
            top_seasonal = seasonal_comparison.head(5)
            
            return {
                'type': 'seasonal_overview',
                'title': 'Seasonal Trends Overview',
                'top_seasonal_items': top_seasonal.to_dict('records')
            }
    
    def _handle_top_products_query(self, query: str) -> Dict:
        """Handle queries about top products."""
        if self.data_processor.data is None:
            return {
                'type': 'error',
                'message': 'No data available. Please upload transaction data first.',
                'suggestion': 'Upload a CSV file with transaction data.'
            }
        
        top_items = self.data_processor.get_top_items(10)
        
        return {
            'type': 'top_products',
            'title': 'Top 10 Best-Selling Products',
            'products': top_items.to_dict('records')
        }
    
    def _handle_item_association_query(self, query: str) -> Dict:
        """Handle queries about specific item associations."""
        # Extract item name from query
        item_match = re.search(r'(\w+)', query)
        if not item_match:
            return {
                'type': 'error',
                'message': 'Please specify a product name.',
                'suggestion': 'Try asking "What goes with milk?"'
            }
        
        item = item_match.group(1).title()
        
        if self.market_basket_analyzer.rules is None or self.market_basket_analyzer.rules.empty:
            return {
                'type': 'error',
                'message': 'No association rules available.',
                'suggestion': 'Run market basket analysis first.'
            }
        
        associations = self.market_basket_analyzer.get_item_associations(item)
        
        if not associations['as_antecedent'] and not associations['as_consequent']:
            return {
                'type': 'info',
                'message': f'No significant associations found for {item}.',
                'suggestion': 'Try checking spelling or a different product.'
            }
        
        return {
            'type': 'item_associations',
            'title': f'Associations for {item}',
            'item': item,
            'associations': associations
        }
    
    def _handle_recommendation_query(self, query: str) -> Dict:
        """Handle general recommendation queries."""
        # This is a general recommendation handler
        recommendations = [
            "Focus on cross-promotion of frequently bought together items",
            "Consider seasonal promotions based on historical data",
            "Personalize marketing based on customer segments",
            "Optimize store layout based on product associations"
        ]
        
        return {
            'type': 'general_recommendations',
            'title': 'Business Recommendations',
            'recommendations': recommendations
        }
    
    def _handle_metrics_query(self, query: str) -> Dict:
        """Handle queries about metrics and statistics."""
        if self.data_processor.data is None:
            return {
                'type': 'error',
                'message': 'No data available for metrics.',
                'suggestion': 'Upload transaction data first.'
            }
        
        summary_stats = self.data_processor.get_summary_stats()
        
        return {
            'type': 'metrics',
            'title': 'Key Business Metrics',
            'metrics': summary_stats
        }
    
    def _handle_general_query(self, query: str) -> Dict:
        """Handle general queries that don't fit specific categories."""
        help_responses = [
            "I can help you with product combinations, customer segments, seasonal trends, and business metrics.",
            "Try asking: 'What is the best product combo?' or 'Who are my key customer segments?'",
            "I can also provide recommendations for seasonal promotions and store layout optimization."
        ]
        
        return {
            'type': 'help',
            'title': 'How I Can Help',
            'responses': help_responses
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
