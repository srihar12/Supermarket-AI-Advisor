import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import re
from datetime import datetime
import json
from collections import defaultdict

class EnhancedAIChatbot:
    """Enhanced AI chatbot with advanced NLP and comprehensive supermarket business intelligence."""
    
    def __init__(self):
        self.data_processor = None
        self.market_basket_analyzer = None
        self.customer_segmentation = None
        self.seasonal_analyzer = None
        self.conversation_history = []
        self.knowledge_base = self._initialize_knowledge_base()
        self.context_memory = defaultdict(list)
        self.user_preferences = {}
        
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize comprehensive knowledge base for supermarket business intelligence."""
        return {
            'business_terms': {
                'lift': 'Lift measures how much more likely item B is purchased when item A is present. Lift > 1 indicates a strong association.',
                'confidence': 'Confidence measures the reliability of the rule - how often item B appears when item A is present.',
                'support': 'Support measures how frequently an itemset appears in all transactions.',
                'conviction': 'Conviction measures the dependency of the consequent on the antecedent.',
                'market_basket_analysis': 'Market basket analysis identifies products that are frequently purchased together.',
                'customer_segmentation': 'Customer segmentation groups customers based on their purchasing behavior.',
                'seasonal_trends': 'Seasonal trends identify how product popularity changes throughout the year.'
            },
            'business_strategies': {
                'cross_promotion': 'Cross-promotion involves marketing complementary products together to increase sales.',
                'store_layout': 'Store layout optimization places frequently bought-together items in close proximity.',
                'seasonal_promotions': 'Seasonal promotions align marketing efforts with seasonal demand patterns.',
                'customer_targeting': 'Customer targeting creates personalized offers based on segment preferences.',
                'inventory_management': 'Inventory management uses demand patterns to optimize stock levels.'
            },
            'product_categories': {
                'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
                'bakery': ['bread', 'buns', 'cakes', 'pastries', 'croissants'],
                'beverages': ['soda', 'juice', 'water', 'coffee', 'tea', 'beer', 'wine'],
                'snacks': ['chips', 'cookies', 'candy', 'nuts', 'crackers'],
                'produce': ['fruits', 'vegetables', 'salads', 'herbs'],
                'meat': ['chicken', 'beef', 'pork', 'turkey', 'fish'],
                'household': ['diapers', 'baby_wipes', 'cleaning_supplies', 'paper_towels'],
                'seasonal': ['christmas_cookies', 'halloween_candy', 'easter_eggs', 'valentine_candy']
            },
            'question_patterns': {
                'what_is': ['what is', 'what does', 'define', 'explain'],
                'how_to': ['how to', 'how can i', 'how do'],
                'why': ['why', 'reason', 'because'],
                'when': ['when', 'time', 'season', 'month'],
                'which': ['which', 'what are the', 'what kind of'],
                'comparison': ['compare', 'difference', 'versus', 'vs']
            }
        }
    
    def set_analyzers(self, data_processor, market_basket_analyzer, 
                     customer_segmentation, seasonal_analyzer):
        """Set the analyzer instances for the chatbot to use."""
        self.data_processor = data_processor
        self.market_basket_analyzer = market_basket_analyzer
        self.customer_segmentation = customer_segmentation
        self.seasonal_analyzer = seasonal_analyzer
    
    def process_query(self, query: str) -> Dict:
        """Process user query with enhanced NLP and generate intelligent response."""
        query_lower = query.lower()
        
        # Store query in conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'response': None
        })
        
        # Update context memory
        self._update_context_memory(query)
        
        # Enhanced query classification
        query_type = self._classify_query_enhanced(query_lower)
        
        # Generate response based on query type
        response = self._generate_response_enhanced(query, query_type)
        
        # Update conversation history
        self.conversation_history[-1]['response'] = response
        
        return response
    
    def _classify_query_enhanced(self, query: str) -> str:
        """Enhanced query classification with better NLP."""
        # Check for definitions/explanations
        if any(pattern in query for pattern in self.knowledge_base['question_patterns']['what_is']):
            return 'definition'
        
        # Check for how-to questions
        if any(pattern in query for pattern in self.knowledge_base['question_patterns']['how_to']):
            return 'how_to'
        
        # Check for comparison questions
        if any(pattern in query for pattern in self.knowledge_base['question_patterns']['comparison']):
            return 'comparison'
        
        # Check for specific business terms
        for term, definition in self.knowledge_base['business_terms'].items():
            if term.replace('_', ' ') in query:
                return 'business_definition'
        
        # Check for product category queries
        for category, items in self.knowledge_base['product_categories'].items():
            if any(item in query for item in items):
                return 'product_category'
        
        # Original query types
        if self._is_product_combo_query(query):
            return 'product_combos'
        elif self._is_customer_segment_query(query):
            return 'customer_segments'
        elif self._is_seasonal_query(query):
            return 'seasonal_trends'
        elif self._is_top_products_query(query):
            return 'top_products'
        elif self._is_item_association_query(query):
            return 'item_associations'
        elif self._is_recommendation_query(query):
            return 'recommendations'
        elif self._is_metrics_query(query):
            return 'metrics'
        else:
            return 'general'
    
    def _generate_response_enhanced(self, query: str, query_type: str) -> Dict:
        """Generate enhanced response based on query type."""
        try:
            if query_type == 'definition':
                return self._handle_definition_query(query)
            elif query_type == 'how_to':
                return self._handle_how_to_query(query)
            elif query_type == 'comparison':
                return self._handle_comparison_query(query)
            elif query_type == 'business_definition':
                return self._handle_business_definition_query(query)
            elif query_type == 'product_category':
                return self._handle_product_category_query(query)
            elif query_type == 'product_combos':
                return self._handle_product_combo_query_enhanced(query)
            elif query_type == 'customer_segments':
                return self._handle_customer_segment_query_enhanced(query)
            elif query_type == 'seasonal_trends':
                return self._handle_seasonal_query_enhanced(query)
            elif query_type == 'top_products':
                return self._handle_top_products_query_enhanced(query)
            elif query_type == 'item_associations':
                return self._handle_item_association_query_enhanced(query)
            elif query_type == 'recommendations':
                return self._handle_recommendation_query_enhanced(query)
            elif query_type == 'metrics':
                return self._handle_metrics_query_enhanced(query)
            else:
                return self._handle_general_query_enhanced(query)
        except Exception as e:
            return {
                'type': 'error',
                'message': f'I encountered an error while processing your question: {str(e)}',
                'suggestion': 'Please try rephrasing your question or check if the data analysis has been completed.'
            }
    
    def _handle_definition_query(self, query: str) -> Dict:
        """Handle definition/explanation queries."""
        # Extract the term being defined
        term_match = re.search(r'what is (?:the )?([^?]+)', query.lower())
        if term_match:
            term = term_match.group(1).strip()
            
            # Check business terms
            for business_term, definition in self.knowledge_base['business_terms'].items():
                if term.replace(' ', '_') == business_term or term.replace(' ', '_') in business_term:
                    return {
                        'type': 'definition',
                        'term': term,
                        'definition': definition,
                        'examples': self._get_examples_for_term(business_term),
                        'title': f'Understanding {term.title()}'
                    }
            
            # Check business strategies
            for strategy, description in self.knowledge_base['business_strategies'].items():
                if term.replace(' ', '_') == strategy or term.replace(' ', '_') in strategy:
                    return {
                        'type': 'definition',
                        'term': term,
                        'definition': description,
                        'examples': self._get_examples_for_strategy(strategy),
                        'title': f'Understanding {term.title()}'
                    }
        
        return {
            'type': 'definition',
            'term': 'Unknown',
            'definition': 'I don\'t have a specific definition for that term. Could you be more specific?',
            'suggestion': 'Try asking about business terms like "lift", "confidence", or "market basket analysis"',
            'title': 'Definition Request'
        }
    
    def _handle_how_to_query(self, query: str) -> Dict:
        """Handle how-to questions with step-by-step guidance."""
        if 'store layout' in query:
            return {
                'type': 'how_to',
                'title': 'How to Optimize Store Layout',
                'steps': [
                    '1. Analyze market basket results to find strong product associations',
                    '2. Place high-lift items (lift > 2) in close proximity',
                    '3. Position complementary items at eye level',
                    '4. Create themed sections (e.g., "Breakfast Zone" with milk, bread, cereal)',
                    '5. Place impulse items near checkout based on association rules',
                    '6. Monitor sales changes after layout adjustments'
                ],
                'benefits': ['Increased cross-selling', 'Better customer experience', 'Higher average transaction value']
            }
        
        elif 'increase sales' in query:
            return {
                'type': 'how_to',
                'title': 'How to Increase Sales',
                'strategies': [
                    'Cross-promotion: Bundle frequently bought-together items',
                    'Seasonal promotions: Align with seasonal demand patterns',
                    'Customer targeting: Personalize offers for different segments',
                    'Store optimization: Improve layout based on product associations',
                    'Inventory management: Ensure popular items are always in stock'
                ],
                'implementation': 'Start with the highest-lift associations and measure the impact before expanding.'
            }
        
        elif 'customer segmentation' in query:
            return {
                'type': 'how_to',
                'title': 'How to Use Customer Segmentation',
                'steps': [
                    '1. Analyze customer segments to understand purchasing patterns',
                    '2. Create targeted marketing campaigns for each segment',
                    '3. Develop personalized promotions based on segment preferences',
                    '4. Adjust product assortment for different customer types',
                    '5. Measure campaign effectiveness by segment'
                ],
                'examples': 'Family shoppers might respond to bulk discounts, while quick shoppers prefer convenience bundles.'
            }
        
        return {
            'type': 'how_to',
            'title': 'General Guidance',
            'message': 'I can help you with specific strategies. Try asking about store layout, increasing sales, or customer segmentation.',
            'suggestion': 'Be more specific about what you\'d like to improve in your supermarket business.'
        }
    
    def _handle_comparison_query(self, query: str) -> Dict:
        """Handle comparison questions between different metrics or strategies."""
        if 'lift vs confidence' in query or 'lift and confidence' in query:
            return {
                'type': 'comparison',
                'title': 'Lift vs Confidence: Key Differences',
                'lift': {
                    'definition': 'Measures how much more likely item B is purchased when item A is present',
                    'interpretation': 'Lift > 1 = Strong association, Lift = 1 = No relationship',
                    'use_case': 'Best for identifying meaningful product relationships'
                },
                'confidence': {
                    'definition': 'Measures how reliably item B appears when item A is present',
                    'interpretation': 'Higher confidence = More reliable rule',
                    'use_case': 'Best for practical implementation decisions'
                },
                'recommendation': 'Use both metrics together: High lift + High confidence = Best opportunities'
            }
        
        elif 'customer segments' in query:
            return {
                'type': 'comparison',
                'title': 'Customer Segment Comparison',
                'message': 'I can compare different customer segments once the analysis is complete.',
                'suggestion': 'Run the customer segmentation analysis first, then ask for segment comparisons.'
            }
        
        return {
            'type': 'comparison',
            'title': 'Comparison Analysis',
            'message': 'I can help you compare different metrics, strategies, or segments.',
            'suggestion': 'Try asking about "lift vs confidence" or specific customer segments.'
        }
    
    def _handle_business_definition_query(self, query: str) -> Dict:
        """Handle business term definitions with examples."""
        for term, definition in self.knowledge_base['business_terms'].items():
            if term.replace('_', ' ') in query:
                return {
                    'type': 'business_definition',
                    'term': term.replace('_', ' ').title(),
                    'definition': definition,
                    'examples': self._get_examples_for_term(term),
                    'practical_application': self._get_practical_application(term),
                    'title': f'Business Intelligence: {term.replace("_", " ").title()}'
                }
        
        return self._handle_definition_query(query)
    
    def _handle_product_category_query(self, query: str) -> Dict:
        """Handle queries about specific product categories."""
        for category, items in self.knowledge_base['product_categories'].items():
            if any(item in query for item in items):
                return {
                    'type': 'product_category',
                    'category': category.title(),
                    'items_in_category': items,
                    'insights': self._get_category_insights(category),
                    'recommendations': self._get_category_recommendations(category),
                    'title': f'{category.title()} Category Analysis'
                }
        
        return {
            'type': 'product_category',
            'title': 'Product Category Analysis',
            'message': 'I can analyze specific product categories once you upload data.',
            'suggestion': 'Upload transaction data and run analysis to get category-specific insights.'
        }
    
    def _handle_product_combo_query_enhanced(self, query: str) -> Dict:
        """Enhanced product combination query handler."""
        self._ensure_association_rules()
        
        if self.market_basket_analyzer.rules is None or self.market_basket_analyzer.rules.empty:
            return {
                'type': 'error',
                'message': 'No association rules available. Please run market basket analysis first.',
                'suggestion': 'Upload your transaction data and click "Run Analysis" to generate product combinations.',
                'help_text': 'Market basket analysis requires transaction data with product names and transaction IDs.'
            }
        
        # Get top rules with more comprehensive analysis
        top_rules = self.market_basket_analyzer.get_top_rules(10, 'lift')
        
        if top_rules.empty:
            return {
                'type': 'info',
                'message': 'No significant product combinations found in your data.',
                'suggestions': [
                    'Try lowering the minimum support threshold',
                    'Check if you have enough transaction data',
                    'Verify product names are consistent'
                ],
                'title': 'Product Combination Analysis'
            }
        
        # Enhanced combo analysis
        combos = []
        insights = []
        
        for _, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            
            combo_data = {
                'combo': f"{antecedents} → {consequents}",
                'lift': rule['lift'],
                'confidence': f"{rule['confidence']:.1%}",
                'support': f"{rule['support']:.1%}",
                'strength': self._classify_rule_strength(rule['lift'], rule['confidence']),
                'action': self._generate_action_recommendation(antecedents, consequents, rule['lift']),
                'explanation': f"When customers buy {antecedents}, they are {rule['lift']:.1f}x more likely to buy {consequents}. This occurs in {rule['confidence']:.1%} of such cases."
            }
            combos.append(combo_data)
            
            # Generate insights
            if rule['lift'] > 3:
                insights.append(f"Strong association between {antecedents} and {consequents}")
        
        return {
            'type': 'product_combos_enhanced',
            'title': '🎯 Advanced Product Combination Analysis',
            'combos': combos,
            'insights': insights,
            'total_rules': len(self.market_basket_analyzer.rules),
            'recommendations': self._generate_combo_recommendations(top_rules)
        }
    
    def _handle_customer_segment_query_enhanced(self, query: str) -> Dict:
        """Enhanced customer segment query handler."""
        if self.customer_segmentation.cluster_labels is None:
            return {
                'type': 'error',
                'message': 'No customer segmentation available. Please run customer analysis first.',
                'suggestion': 'Upload transaction data and run the complete analysis to identify customer segments.',
                'help_text': 'Customer segmentation groups shoppers based on their purchasing patterns.'
            }
        
        cluster_summary = self.customer_segmentation.get_cluster_summary()
        cluster_analysis = self.customer_segmentation.analyze_clusters(
            self.data_processor.data,
            self.data_processor.transaction_col,
            self.data_processor.item_col
        )
        
        if cluster_summary.empty:
            return {
                'type': 'info',
                'message': 'No customer segments identified in your data.',
                'suggestions': [
                    'Check if you have sufficient transaction data',
                    'Verify customer identification in your data',
                    'Consider adjusting clustering parameters'
                ]
            }
        
        # Enhanced segment analysis
        segments = []
        for cluster_id, analysis in cluster_analysis.items():
            segment_data = {
                'segment_id': cluster_id,
                'segment_name': analysis['segment_name'],
                'customer_count': analysis['customer_count'],
                'percentage': f"{(analysis['customer_count'] / len(self.customer_segmentation.customer_features) * 100):.1f}%",
                'avg_items_per_customer': f"{analysis['avg_items_per_customer']:.1f}",
                'total_transactions': analysis['total_transactions'],
                'unique_items': analysis['unique_items_purchased'],
                'top_items': analysis['top_items'],
                'characteristics': self._generate_segment_characteristics(analysis),
                'marketing_suggestions': self._generate_marketing_suggestions(analysis['segment_name'], analysis['top_items'])
            }
            segments.append(segment_data)
        
        return {
            'type': 'customer_segments_enhanced',
            'title': '👥 Advanced Customer Segmentation Analysis',
            'segments': segments,
            'total_customers': len(self.customer_segmentation.customer_features),
            'insights': self._generate_segment_insights(segments),
            'strategic_recommendations': self._generate_segment_strategies(segments)
        }
    
    def _handle_seasonal_query_enhanced(self, query: str) -> Dict:
        """Enhanced seasonal trend query handler."""
        if self.seasonal_analyzer.seasonal_patterns is None:
            return {
                'type': 'error',
                'message': 'No seasonal analysis available. Please ensure your data includes date information.',
                'suggestion': 'Upload transaction data with date columns and run the complete analysis.',
                'help_text': 'Seasonal analysis requires transaction dates to identify patterns throughout the year.'
            }
        
        # Enhanced seasonal analysis
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
            recommendations = self.seasonal_analyzer.get_seasonal_recommendations(target_season, 15)
            seasonal_insights = self._generate_seasonal_insights(target_season)
            
            return {
                'type': 'seasonal_recommendations_enhanced',
                'title': f'☀️ Advanced {target_season} Seasonal Analysis',
                'season': target_season,
                'recommendations': recommendations,
                'insights': seasonal_insights,
                'action_plan': self._generate_seasonal_action_plan(target_season, recommendations)
            }
        else:
            # Comprehensive seasonal overview
            seasonal_comparison = self.seasonal_analyzer.get_seasonal_comparison()
            year_round_insights = self._generate_year_round_insights(seasonal_comparison)
            
            return {
                'type': 'seasonal_overview_enhanced',
                'title': '📅 Comprehensive Seasonal Trends Analysis',
                'seasonal_comparison': seasonal_comparison.head(10).to_dict('records'),
                'insights': year_round_insights,
                'year_round_strategy': self._generate_year_round_strategy(seasonal_comparison)
            }
    
    def _handle_top_products_query_enhanced(self, query: str) -> Dict:
        """Enhanced top products query handler."""
        if self.data_processor.data is None:
            return {
                'type': 'error',
                'message': 'No data available. Please upload transaction data first.',
                'suggestion': 'Upload a CSV file with transaction data including product names and transaction IDs.'
            }
        
        top_items = self.data_processor.get_top_items(20)
        
        # Enhanced analysis
        insights = []
        if len(top_items) >= 5:
            top_5_revenue = top_items.head(5)['Count'].sum()
            total_revenue = top_items['Count'].sum()
            insights.append(f"Top 5 products represent {(top_5_revenue/total_revenue*100):.1f}% of total purchases")
        
        # Category analysis
        category_performance = self._analyze_category_performance(top_items)
        
        return {
            'type': 'top_products_enhanced',
            'title': '🏆 Advanced Top Products Analysis',
            'products': top_items.to_dict('records'),
            'insights': insights,
            'category_performance': category_performance,
            'recommendations': self._generate_product_recommendations(top_items)
        }
    
    def _handle_item_association_query_enhanced(self, query: str) -> Dict:
        """Enhanced item association query handler."""
        # Better item extraction
        item_patterns = [
            r'what goes with ([^?]+)',
            r'what should i put with ([^?]+)',
            r'association with ([^?]+)',
            r'([a-zA-Z\s]+) associations?',
            r'combinations with ([^?]+)'
        ]
        
        item = None
        for pattern in item_patterns:
            match = re.search(pattern, query.lower())
            if match:
                item = match.group(1).strip().title()
                break
        
        if not item:
            return {
                'type': 'error',
                'message': 'Please specify a product name for association analysis.',
                'examples': ['Try: "What goes with milk?"', 'Try: "Associations with bread"', 'Try: "What should I put with chips?"'],
                'title': 'Item Association Analysis'
            }
        
        self._ensure_association_rules()
        
        if self.market_basket_analyzer.rules is None or self.market_basket_analyzer.rules.empty:
            return {
                'type': 'error',
                'message': 'No association rules available.',
                'suggestion': 'Run market basket analysis first to discover product relationships.'
            }
        
        associations = self.market_basket_analyzer.get_item_associations(item)
        
        if not associations['as_antecedent'] and not associations['as_consequent']:
            # Try partial matches
            similar_items = self._find_similar_items(item)
            if similar_items:
                return {
                    'type': 'suggestions',
                    'item': item,
                    'similar_items': similar_items,
                    'message': f'No direct associations found for {item}, but here are similar items:',
                    'title': f'Association Analysis for {item}'
                }
            
            return {
                'type': 'info',
                'message': f'No significant associations found for {item}.',
                'suggestions': [
                    'Check the spelling of the product name',
                    'Ensure the product exists in your transaction data',
                    'Try a more general product category'
                ],
                'title': f'Association Analysis for {item}'
            }
        
        # Enhanced association analysis
        analysis = {
            'item': item,
            'as_trigger': associations['as_antecedent'][:5],  # Top 5
            'as_result': associations['as_consequent'][:5],  # Top 5
            'insights': self._generate_association_insights(item, associations),
            'recommendations': self._generate_association_recommendations(item, associations)
        }
        
        return {
            'type': 'item_associations_enhanced',
            'title': f'🔗 Advanced Association Analysis for {item}',
            'analysis': analysis
        }
    
    def _handle_recommendation_query_enhanced(self, query: str) -> Dict:
        """Enhanced recommendation query handler."""
        recommendations = self._generate_comprehensive_recommendations(query)
        
        return {
            'type': 'recommendations_enhanced',
            'title': '💡 Advanced Business Recommendations',
            'recommendations': recommendations,
            'implementation_priority': self._prioritize_recommendations(recommendations),
            'expected_outcomes': self._generate_expected_outcomes(recommendations)
        }
    
    def _handle_metrics_query_enhanced(self, query: str) -> Dict:
        """Enhanced metrics query handler."""
        if self.data_processor.data is None:
            return {
                'type': 'error',
                'message': 'No data available for metrics analysis.',
                'suggestion': 'Upload transaction data first to generate comprehensive business metrics.'
            }
        
        summary_stats = self.data_processor.get_summary_stats()
        mb_metrics = self.market_basket_analyzer.get_summary_metrics()
        
        # Enhanced metrics analysis
        performance_analysis = self._analyze_performance(summary_stats, mb_metrics)
        trends = self._identify_trends()
        
        return {
            'type': 'metrics_enhanced',
            'title': '📊 Comprehensive Business Metrics Dashboard',
            'summary_stats': summary_stats,
            'market_basket_metrics': mb_metrics,
            'performance_analysis': performance_analysis,
            'trends': trends,
            'benchmarks': self._generate_benchmarks(summary_stats)
        }
    
    def _handle_general_query_enhanced(self, query: str) -> Dict:
        """Enhanced general query handler with better help and suggestions."""
        capabilities = [
            "🎯 **Product Combinations**: Discover frequently bought-together items",
            "👥 **Customer Segments**: Understand different shopper types and behaviors",
            "☀️ **Seasonal Trends**: Identify seasonal patterns and optimize promotions",
            "📊 **Business Metrics**: Track key performance indicators",
            "🔗 **Item Associations**: Find what products work well together",
            "💡 **Business Strategy**: Get actionable recommendations",
            "📚 **Definitions**: Learn about business intelligence concepts"
        ]
        
        example_questions = [
            "What is the best product combo?",
            "Who are my key customer segments?",
            "What should I promote this summer?",
            "What goes with milk?",
            "How can I increase sales?",
            "What is lift in market basket analysis?",
            "Compare customer segments"
        ]
        
        return {
            'type': 'help_enhanced',
            'title': '🤖 Supermarket AI Advisor - Capabilities',
            'capabilities': capabilities,
            'example_questions': example_questions,
            'getting_started': [
                "1. Upload your transaction data (CSV format)",
                "2. Map Transaction ID and Item Name columns", 
                "3. Adjust analysis parameters if needed",
                "4. Click 'Run Analysis' to generate insights",
                "5. Ask me questions about your results!"
            ],
            'pro_tips': [
                "Use specific product names for better association analysis",
                "Include date columns for seasonal trend analysis",
                "Try different question formats to explore various insights"
            ]
        }
    
    # Helper methods for enhanced responses
    def _get_examples_for_term(self, term: str) -> List[str]:
        """Get practical examples for business terms."""
        examples = {
            'lift': ["Milk → Bread with lift 2.5 means bread sales are 2.5x higher when milk is purchased"],
            'confidence': ["If 80% of customers who buy milk also buy bread, confidence = 0.8"],
            'support': ["If milk appears in 10% of all transactions, support = 0.1"],
            'market_basket_analysis': ["Discovering that diapers and baby wipes are frequently bought together"],
            'customer_segmentation': ["Identifying 'Family Shoppers' vs 'Quick Shoppers' based on purchasing patterns"]
        }
        return examples.get(term, ["Contact support for specific examples"])
    
    def _get_examples_for_strategy(self, strategy: str) -> List[str]:
        """Get examples for business strategies."""
        examples = {
            'cross_promotion': ["Buy cereal, get 10% off milk", "Chips and soda combo deal"],
            'store_layout': ["Place milk and bread in adjacent aisles", "Create seasonal display areas"],
            'seasonal_promotions': ["Summer BBQ bundles", "Holiday baking essentials"]
        }
        return examples.get(strategy, ["Contact support for strategy examples"])
    
    def _get_practical_application(self, term: str) -> str:
        """Get practical application for business terms."""
        applications = {
            'lift': "Use lift > 2.0 to identify strong product relationships for store layout optimization",
            'confidence': "Use confidence > 0.7 for reliable cross-promotion opportunities",
            'support': "Use support > 0.05 to ensure recommendations apply to enough customers"
        }
        return applications.get(term, "Apply this metric to improve business decisions")
    
    def _classify_rule_strength(self, lift: float, confidence: float) -> str:
        """Classify the strength of association rules."""
        if lift > 3 and confidence > 0.7:
            return "Very Strong"
        elif lift > 2 and confidence > 0.5:
            return "Strong"
        elif lift > 1.5 and confidence > 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    def _generate_action_recommendation(self, antecedents: str, consequents: str, lift: float) -> str:
        """Generate specific action recommendations for product pairs."""
        if lift > 3:
            return f"Place {consequents} prominently near {antecedents}"
        elif lift > 2:
            return f"Create bundle deal for {antecedents} + {consequents}"
        else:
            return f"Consider cross-promotion of {antecedents} and {consequents}"
    
    def _get_examples_for_term(self, term: str) -> List[str]:
        """Get practical examples for business terms."""
        return self._get_examples_for_term(term)
    
    def _get_practical_application(self, term: str) -> str:
        """Get practical application for business terms."""
        return self._get_practical_application(term)
    
    def _generate_combo_recommendations(self, top_rules: pd.DataFrame) -> List[str]:
        """Generate strategic recommendations based on top combinations."""
        recommendations = []
        
        if len(top_rules) > 0:
            avg_lift = top_rules['lift'].mean()
            if avg_lift > 2:
                recommendations.append("Strong product associations detected - optimize store layout")
            
            high_confidence_rules = top_rules[top_rules['confidence'] > 0.7]
            if len(high_confidence_rules) > 0:
                recommendations.append("High-confidence rules available - create bundle promotions")
        
        return recommendations
    
    def _generate_segment_characteristics(self, analysis: Dict) -> str:
        """Generate detailed segment characteristics."""
        characteristics = []
        
        if analysis['avg_items_per_customer'] > 10:
            characteristics.append("High-volume shoppers")
        elif analysis['avg_items_per_customer'] < 5:
            characteristics.append("Quick/convenience shoppers")
        else:
            characteristics.append("Regular shoppers")
        
        if analysis['unique_items_purchased'] > 20:
            characteristics.append("Diverse product preferences")
        
        return ", ".join(characteristics)
    
    def _generate_marketing_suggestions(self, segment_name: str, top_items: Dict) -> List[str]:
        """Generate marketing suggestions for customer segments."""
        suggestions = []
        
        if "Family" in segment_name:
            suggestions.extend([
                "Family bundle promotions",
                "Bulk purchase discounts",
                "Weekly meal planning suggestions"
            ])
        elif "Quick" in segment_name:
            suggestions.extend([
                "Express checkout promotions",
                "Ready-to-eat meal deals",
                "Convenience product placement"
            ])
        elif "Snack" in segment_name:
            suggestions.extend([
                "Impulse buy displays",
                "Snack combo deals",
                "Evening promotion timing"
            ])
        
        return suggestions
    
    def _generate_segment_insights(self, segments: List[Dict]) -> List[str]:
        """Generate insights from customer segment analysis."""
        insights = []
        
        if len(segments) > 0:
            largest_segment = max(segments, key=lambda x: int(x['customer_count']))
            insights.append(f"Largest segment: {largest_segment['segment_name']} ({largest_segment['percentage']} of customers)")
        
        high_value_segments = [s for s in segments if float(s['avg_items_per_customer']) > 10]
        if high_value_segments:
            insights.append(f"{len(high_value_segments)} high-value segments identified")
        
        return insights
    
    def _generate_segment_strategies(self, segments: List[Dict]) -> List[str]:
        """Generate strategic recommendations for customer segments."""
        strategies = [
            "Develop targeted marketing campaigns for each segment",
            "Personalize product recommendations based on segment preferences",
            "Optimize inventory based on segment demand patterns",
            "Create segment-specific promotions and bundles"
        ]
        return strategies
    
    def _generate_seasonal_insights(self, season: str) -> List[str]:
        """Generate insights for specific seasons."""
        insights = {
            'Summer': ["Focus on BBQ and outdoor products", "Increase cold beverage inventory"],
            'Winter': ["Promote comfort foods and baking supplies", "Stock up on holiday items"],
            'Spring': ["Emphasize fresh produce and healthy options", "Prepare for Easter items"],
            'Fall': ["Focus on back-to-school and Thanksgiving items", "Prepare comfort food inventory"]
        }
        return insights.get(season, ["Analyze seasonal patterns for this season"])
    
    def _generate_seasonal_action_plan(self, season: str, recommendations: List[Dict]) -> List[str]:
        """Generate action plan for seasonal promotions."""
        action_plan = [
            f"Plan {season} promotions 2 months in advance",
            "Adjust inventory based on seasonal demand",
            "Create seasonal product bundles",
            "Train staff on seasonal product placement"
        ]
        return action_plan
    
    def _generate_year_round_insights(self, seasonal_comparison: pd.DataFrame) -> List[str]:
        """Generate year-round insights from seasonal data."""
        insights = []
        
        if not seasonal_comparison.empty:
            most_seasonal = seasonal_comparison.iloc[0]
            insights.append(f"Most seasonal item: {most_seasonal['item']} (peaks in {most_seasonal['peak_season']})")
        
        return insights
    
    def _generate_year_round_strategy(self, seasonal_comparison: pd.DataFrame) -> List[str]:
        """Generate year-round strategy recommendations."""
        return [
            "Develop seasonal calendar for promotions",
            "Plan inventory rotations based on seasonal patterns",
            "Create seasonal display areas in store",
            "Align marketing campaigns with seasonal demand"
        ]
    
    def _analyze_category_performance(self, top_items: pd.DataFrame) -> Dict:
        """Analyze performance by product category."""
        category_performance = {}
        
        for category, items in self.knowledge_base['product_categories'].items():
            category_items = top_items[top_items['Item'].str.contains('|'.join(items), case=False, na=False)]
            if not category_items.empty:
                category_performance[category] = {
                    'count': len(category_items),
                    'total_purchases': category_items['Count'].sum(),
                    'top_item': category_items.iloc[0]['Item'] if not category_items.empty else None
                }
        
        return category_performance
    
    def _generate_product_recommendations(self, top_items: pd.DataFrame) -> List[str]:
        """Generate recommendations based on top products."""
        recommendations = [
            "Ensure top-selling items are always in stock",
            "Place top items in high-traffic areas",
            "Create promotions around top items",
            "Analyze why top items perform well"
        ]
        return recommendations
    
    def _find_similar_items(self, item: str) -> List[str]:
        """Find similar items when exact match not found."""
        similar = []
        item_lower = item.lower()
        
        for category, items in self.knowledge_base['product_categories'].items():
            for category_item in items:
                if item_lower in category_item.lower() or category_item.lower() in item_lower:
                    similar.append(category_item.title())
        
        return similar[:5]  # Return top 5 similar items
    
    def _generate_association_insights(self, item: str, associations: Dict) -> List[str]:
        """Generate insights from item associations."""
        insights = []
        
        if associations['as_antecedent']:
            insights.append(f"{item} strongly influences purchase of other items")
        
        if associations['as_consequent']:
            insights.append(f"{item} is frequently purchased with other items")
        
        return insights
    
    def _generate_association_recommendations(self, item: str, associations: Dict) -> List[str]:
        """Generate recommendations based on item associations."""
        recommendations = []
        
        if associations['as_antecedent']:
            top_consequent = associations['as_antecedent'][0]
            recommendations.append(f"Cross-promote {item} with {top_consequent.get('consequents_str', 'related items')}")
        
        if associations['as_consequent']:
            top_antecedent = associations['as_consequent'][0]
            recommendations.append(f"Place {item} near {top_antecedent.get('antecedents_str', 'trigger items')}")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self, query: str) -> List[Dict]:
        """Generate comprehensive business recommendations."""
        recommendations = [
            {
                'category': 'Store Operations',
                'priority': 'High',
                'action': 'Optimize store layout based on product associations',
                'impact': 'Increased cross-selling and customer satisfaction',
                'implementation': 'Review market basket analysis results and rearrange high-lift item pairs'
            },
            {
                'category': 'Marketing',
                'priority': 'High', 
                'action': 'Create targeted promotions for customer segments',
                'impact': 'Higher conversion rates and customer loyalty',
                'implementation': 'Develop segment-specific campaigns and personalized offers'
            },
            {
                'category': 'Inventory',
                'priority': 'Medium',
                'action': 'Align inventory with seasonal demand patterns',
                'impact': 'Reduced stockouts and improved turnover',
                'implementation': 'Use seasonal analysis to plan inventory rotations'
            },
            {
                'category': 'Customer Experience',
                'priority': 'Medium',
                'action': 'Implement bundle deals for frequently bought-together items',
                'impact': 'Higher average transaction value',
                'implementation': 'Identify top associations and create attractive bundles'
            }
        ]
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> Dict:
        """Prioritize recommendations by impact and ease of implementation."""
        prioritized = {
            'immediate': [r for r in recommendations if r['priority'] == 'High'],
            'short_term': [r for r in recommendations if r['priority'] == 'Medium'],
            'long_term': [r for r in recommendations if r['priority'] == 'Low']
        }
        return prioritized
    
    def _generate_expected_outcomes(self, recommendations: List[Dict]) -> List[str]:
        """Generate expected outcomes from implementing recommendations."""
        outcomes = [
            "10-15% increase in cross-selling revenue",
            "5-10% improvement in customer satisfaction",
            "20% reduction in stockouts for seasonal items",
            "Higher customer retention through personalized marketing"
        ]
        return outcomes
    
    def _analyze_performance(self, summary_stats: Dict, mb_metrics: Dict) -> Dict:
        """Analyze overall business performance."""
        performance = {
            'transaction_health': 'Good' if summary_stats['total_transactions'] > 100 else 'Needs improvement',
            'product_diversity': 'High' if summary_stats['total_items'] > 50 else 'Moderate',
            'association_strength': 'Strong' if mb_metrics.get('avg_lift', 0) > 2 else 'Moderate',
            'data_quality': 'Good' if summary_stats['avg_items_per_transaction'] > 2 else 'Review needed'
        }
        return performance
    
    def _identify_trends(self) -> List[str]:
        """Identify business trends from data."""
        trends = [
            "Analyzing purchasing patterns for trend identification",
            "Monitor seasonal changes in product popularity",
            "Track customer segment evolution over time"
        ]
        return trends
    
    def _generate_benchmarks(self, summary_stats: Dict) -> Dict:
        """Generate industry benchmarks for comparison."""
        benchmarks = {
            'avg_transaction_size': {
                'current': summary_stats['avg_items_per_transaction'],
                'industry_average': 3.5,
                'assessment': 'Above average' if summary_stats['avg_items_per_transaction'] > 3.5 else 'Below average'
            },
            'product_variety': {
                'current': summary_stats['total_items'],
                'industry_average': 100,
                'assessment': 'Good variety' if summary_stats['total_items'] > 80 else 'Limited variety'
            }
        }
        return benchmarks
    
    def _update_context_memory(self, query: str) -> None:
        """Update context memory for better conversation flow."""
        # Store recent queries for context
        self.context_memory['recent_queries'].append(query)
        
        # Keep only last 5 queries
        if len(self.context_memory['recent_queries']) > 5:
            self.context_memory['recent_queries'].pop(0)
    
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
    
    # Original query detection methods (kept for compatibility)
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
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.context_memory.clear()
    
    def train_with_feedback(self, query: str, feedback: str, rating: int) -> None:
        """Train the chatbot with user feedback."""
        # Store feedback for future improvements
        self.context_memory['feedback'].append({
            'query': query,
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.now()
        })
    
    def get_training_progress(self) -> Dict:
        """Get training progress and statistics."""
        feedback_count = len(self.context_memory.get('feedback', []))
        avg_rating = 0
        
        if feedback_count > 0:
            ratings = [f['rating'] for f in self.context_memory['feedback']]
            avg_rating = sum(ratings) / len(ratings)
        
        return {
            'total_interactions': len(self.conversation_history),
            'feedback_received': feedback_count,
            'average_rating': avg_rating,
            'improvement_areas': self._identify_improvement_areas()
        }
    
    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas for improvement based on feedback."""
        areas = []
        
        feedback_list = self.context_memory.get('feedback', [])
        if feedback_list:
            low_ratings = [f for f in feedback_list if f['rating'] < 3]
            if len(low_ratings) > len(feedback_list) * 0.3:
                areas.append("Response accuracy needs improvement")
            
            # Analyze feedback patterns
            for feedback_item in low_ratings:
                if 'unclear' in feedback_item['feedback'].lower():
                    areas.append("Need clearer explanations")
                if 'wrong' in feedback_item['feedback'].lower():
                    areas.append("Improve answer accuracy")
        
        return areas
    
    def _get_category_insights(self, category: str) -> List[str]:
        """Get insights for specific product categories."""
        insights = {
            'dairy': [
                "Dairy products are frequently purchased with bread and cereal",
                "Milk is one of the most commonly purchased items",
                "Dairy items show consistent demand across all seasons"
            ],
            'bakery': [
                "Bakery items are often purchased with dairy products",
                "Bread shows high purchase frequency",
                "Fresh bakery items may have seasonal demand patterns"
            ],
            'beverages': [
                "Beverages are frequently purchased with snacks",
                "Soda and water show consistent demand",
                "Seasonal beverages show peak demand in summer"
            ],
            'snacks': [
                "Snacks are often impulse purchases",
                "Chips and soda show strong association",
                "Snack items may be purchased during specific times"
            ]
        }
        
        category_lower = category.lower()
        for key, category_insights in insights.items():
            if key in category_lower or any(item in category_lower for item in self.knowledge_base['product_categories'].get(key, [])):
                return category_insights
        
        return ["Analyze purchasing patterns for this category", "Consider cross-promotion opportunities", "Monitor seasonal demand variations"]
    
    def _get_category_recommendations(self, category: str) -> List[str]:
        """Get recommendations for specific product categories."""
        recommendations = {
            'dairy': [
                "Place dairy products near bakery items",
                "Create breakfast bundle promotions",
                "Ensure consistent inventory levels"
            ],
            'bakery': [
                "Position bakery items in high-traffic areas",
                "Create daily fresh promotions",
                "Cross-promote with dairy products"
            ],
            'beverages': [
                "Place beverages near snack items",
                "Create meal combo promotions",
                "Ensure adequate refrigeration"
            ],
            'snacks': [
                "Place snacks near checkout for impulse buys",
                "Create movie night bundles",
                "Position near beverages"
            ]
        }
        
        category_lower = category.lower()
        for key, category_recs in recommendations.items():
            if key in category_lower or any(item in category_lower for item in self.knowledge_base['product_categories'].get(key, [])):
                return category_recs
        
        return ["Analyze customer preferences for this category", "Optimize product placement", "Create targeted promotions"]
    
    def _generate_association_insights(self, item: str, associations: Dict) -> List[str]:
        """Generate insights from item associations."""
        insights = []
        
        if associations['as_antecedent']:
            insights.append(f"{item} strongly influences purchase of other items")
        
        if associations['as_consequent']:
            insights.append(f"{item} is frequently purchased with other items")
        
        # Add more specific insights based on association strength
        if associations['as_antecedent']:
            strong_associations = [a for a in associations['as_antecedent'] if a.get('lift', 0) > 2]
            if strong_associations:
                insights.append(f"{item} has {len(strong_associations)} strong product associations")
        
        return insights
    
    def _generate_association_recommendations(self, item: str, associations: Dict) -> List[str]:
        """Generate recommendations based on item associations."""
        recommendations = []
        
        if associations['as_antecedent']:
            top_consequent = associations['as_antecedent'][0]
            consequent_items = top_consequent.get('consequents_str', 'related items')
            recommendations.append(f"Cross-promote {item} with {consequent_items}")
        
        if associations['as_consequent']:
            top_antecedent = associations['as_consequent'][0]
            antecedent_items = top_antecedent.get('antecedents_str', 'trigger items')
            recommendations.append(f"Place {item} near {antecedent_items}")
        
        # Add general recommendations
        recommendations.extend([
            "Monitor sales impact of placement changes",
            "Consider bundle promotions for strong associations",
            "Track customer response to cross-promotions"
        ])
        
        return recommendations[:3]  # Return top 3 recommendations
