import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our custom modules
from data_processor import DataProcessor
from market_basket_analyzer import MarketBasketAnalyzer
from customer_segmentation import CustomerSegmentation
from seasonal_analyzer import SeasonalAnalyzer
from visualizations import Visualizer
from enhanced_ai_chatbot import EnhancedAIChatbot

# Configure Streamlit page
st.set_page_config(
    page_title="Supermarket AI Advisor 🤖🛒",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling for clean interface
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        font-family: Arial, sans-serif;
    }
    .chat-message {
        padding: 0.5rem;
        margin: 0.3rem 0;
        max-width: 80%;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .user-message {
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'market_basket_analyzer' not in st.session_state:
    st.session_state.market_basket_analyzer = MarketBasketAnalyzer()
if 'customer_segmentation' not in st.session_state:
    st.session_state.customer_segmentation = CustomerSegmentation()
if 'seasonal_analyzer' not in st.session_state:
    st.session_state.seasonal_analyzer = SeasonalAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = EnhancedAIChatbot()
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# Set up chatbot with analyzers
st.session_state.chatbot.set_analyzers(
    st.session_state.data_processor,
    st.session_state.market_basket_analyzer,
    st.session_state.customer_segmentation,
    st.session_state.seasonal_analyzer
)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 Supermarket AI Advisor 🛒</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for data upload and configuration
    with st.sidebar:
        st.header("📊 Data Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with transaction data"
        )
        
        if uploaded_file is not None:
            # Read and store the data
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                data = pd.read_csv(stringio)
                st.session_state.uploaded_data = data
                
                st.success(f"✅ Data loaded successfully! {len(data)} rows found.")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(data.head())
                
                # Column mapping
                st.subheader("🗺️ Column Mapping")
                columns = data.columns.tolist()
                
                transaction_col = st.selectbox(
                    "Transaction ID Column",
                    columns,
                    help="Select the column that contains transaction identifiers"
                )
                
                item_col = st.selectbox(
                    "Item Name Column", 
                    columns,
                    help="Select the column that contains product names"
                )
                
                # Check for date column
                date_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
                date_col = None
                if date_cols:
                    date_col = st.selectbox(
                        "Date Column (Optional)",
                        ['None'] + date_cols,
                        help="Select the column that contains transaction dates for seasonal analysis"
                    )
                    if date_col == 'None':
                        date_col = None
                
                # Analysis parameters
                st.subheader("⚙️ Analysis Parameters")
                
                min_support = st.slider(
                    "Minimum Support (Popularity Filter)",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.001,
                    step=0.001,
                    help="Minimum frequency for item combinations (0.1% = 0.001)",
                    key="min_support"
                )
                
                min_confidence = st.slider(
                    "Minimum Confidence (Reliability Filter)",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Minimum reliability for association rules",
                    key="min_confidence"
                )
                
                # Run analysis button
                if st.button("🚀 Run Analysis", type="primary"):
                    with st.spinner("Running analysis... This may take a moment."):
                        try:
                            # Process data
                            st.session_state.data_processor.load_data(StringIO(uploaded_file.getvalue().decode("utf-8")))
                            st.session_state.data_processor.validate_columns(transaction_col, item_col, date_col)
                            st.session_state.data_processor.preprocess_data()
                            
                            # Get baskets for analysis
                            baskets = st.session_state.data_processor.get_transaction_baskets()
                            
                            # Market basket analysis
                            st.session_state.market_basket_analyzer.prepare_transactions(baskets)
                            st.session_state.market_basket_analyzer.find_frequent_itemsets(min_support)
                            st.session_state.market_basket_analyzer.generate_association_rules(min_confidence)
                            
                            # Customer segmentation
                            st.session_state.customer_segmentation.prepare_customer_features(
                                st.session_state.data_processor.data, transaction_col, item_col
                            )
                            st.session_state.customer_segmentation.perform_clustering()
                            
                            # Seasonal analysis (if date column available)
                            if date_col:
                                st.session_state.seasonal_analyzer.prepare_seasonal_data(
                                    st.session_state.data_processor.data, item_col, date_col
                                )
                                st.session_state.seasonal_analyzer.analyze_seasonal_patterns()
                            
                            st.session_state.analysis_complete = True
                            st.success("✅ Analysis completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.session_state.analysis_complete = False
            
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    # Main content area
    if st.session_state.analysis_complete:
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 AI Chat", "📋 Action Plan", "👥 Customer Segments", 
            "☀️ Seasonal Trends", "📊 Visualizations", "📈 Metrics"
        ])
        
        with tab1:
            render_ai_chat()
        
        with tab2:
            render_action_plan()
        
        with tab3:
            render_customer_segments()
        
        with tab4:
            render_seasonal_trends()
        
        with tab5:
            render_visualizations()
        
        with tab6:
            render_metrics()
    
    else:
        # Welcome message
        st.markdown("""
        ## 🎯 Welcome to Supermarket AI Advisor!
        
        Transform your raw transaction data into actionable business intelligence with our AI-powered analysis tools.
        
        ### 📋 Getting Started:
        1. **Upload Data**: Use the sidebar to upload your CSV transaction file
        2. **Map Columns**: Select which columns contain Transaction ID, Item Name, and optionally Date
        3. **Configure**: Adjust sensitivity parameters for market basket analysis
        4. **Analyze**: Click "Run Analysis" to generate insights
        
        ### 🔍 What You'll Discover:
        - 🤖 **AI Chatbot**: Ask natural language questions about your data
        - 🎯 **Product Combinations**: Discover frequently bought-together items
        - 👥 **Customer Segments**: Understand different shopper types
        - ☀️ **Seasonal Trends**: Identify seasonal product patterns
        - 📊 **Interactive Charts**: Visualize data relationships
        - 📈 **Business Metrics**: Track key performance indicators
        
        ### 📊 Data Format Requirements:
        - CSV file with transaction data
        - Transaction ID column (unique identifiers)
        - Item Name column (product names)
        - Date column (optional, for seasonal analysis)
        
        **Ready to get started? Upload your data in the sidebar!** 🚀
        """)

def render_ai_chat():
    st.header("🤖 AI Chatbot - Chitti")
    st.markdown("Ask natural language questions about your supermarket data!")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Chitti:** {message['content']}")
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your supermarket data...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Process query with chatbot
        response = st.session_state.chatbot.process_query(user_input)
        
        # Format response
        bot_response = format_chatbot_response(response)
        
        # Add bot response to history
        st.session_state.chat_history.append({'role': 'bot', 'content': bot_response})
        
        # Rerun to display the new message
        st.rerun()
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What is the best product combo?",
            "Who are my key customer segments?",
            "What should I promote this summer?",
            "What are the top-selling products?",
            "What goes with milk?",
            "How many transactions do I have?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.chat_history.append({'role': 'user', 'content': question})
                response = st.session_state.chatbot.process_query(question)
                bot_response = format_chatbot_response(response)
                st.session_state.chat_history.append({'role': 'bot', 'content': bot_response})
                st.rerun()

def format_chatbot_response(response):
    """Format enhanced chatbot response for display."""
    if response['type'] == 'error':
        suggestion = response.get('suggestion', 'Try rephrasing your question.')
        return f"Error: {response['message']}\n\nSuggestion: {suggestion}"
    
    # Enhanced product combos response
    elif response['type'] in ['product_combos', 'product_combos_enhanced']:
        if 'combos' not in response:
            return "Error: No product combinations found. Please run the market basket analysis first."
        
        combos_text = []
        for combo in response['combos']:
            if 'strength' in combo:
                combos_text.append(f"- {combo['combo']} ({combo['strength']} - Lift: {combo['lift']:.2f}, Confidence: {combo['confidence']})")
            else:
                combos_text.append(f"- {combo['combo']} (Lift: {combo['lift']:.2f}, Confidence: {combo['confidence']})")
            combos_text.append(f"  {combo['explanation']}")
            if 'action' in combo:
                combos_text.append(f"  Action: {combo['action']}")
        
        result = f"{response['title']}\n\n" + "\n".join(combos_text)
        
        if 'insights' in response and response['insights']:
            result += f"\n\nKey Insights:\n" + "\n".join([f"- {insight}" for insight in response['insights']])
        
        if 'recommendations' in response and response['recommendations']:
            result += f"\n\nStrategic Recommendations:\n" + "\n".join([f"- {rec}" for rec in response['recommendations']])
        
        return result
    
    # Enhanced customer segments response
    elif response['type'] in ['customer_segments', 'customer_segments_enhanced']:
        if 'segments' not in response:
            return "Error: No customer segments found. Please run the customer segmentation analysis first."
        
        segments_text = []
        for segment in response['segments']:
            segment_name = segment.get('segment_name', f"Segment {segment['segment_id']}")
            segments_text.append(f"- {segment_name}: {segment['percentage']} of customers")
            if 'characteristics' in segment:
                segments_text.append(f"  {segment['characteristics']}")
            if 'marketing_suggestions' in segment and segment['marketing_suggestions']:
                segments_text.append(f"  Marketing: {', '.join(segment['marketing_suggestions'][:2])}")
        
        result = f"{response['title']}\n\n" + "\n".join(segments_text)
        
        if 'insights' in response and response['insights']:
            result += f"\n\nKey Insights:\n" + "\n".join([f"- {insight}" for insight in response['insights']])
        
        if 'strategic_recommendations' in response and response['strategic_recommendations']:
            result += f"\n\nStrategic Recommendations:\n" + "\n".join([f"- {rec}" for rec in response['strategic_recommendations']])
        
        return result
    
    # Enhanced seasonal recommendations response
    elif response['type'] in ['seasonal_recommendations', 'seasonal_recommendations_enhanced']:
        if 'recommendations' not in response:
            return "Error: No seasonal recommendations found. Please run the seasonal analysis first."
        
        recs_text = []
        for rec in response['recommendations']:
            priority_marker = "HIGH" if rec.get('priority') == 'High' else "MEDIUM"
            recs_text.append(f"- {rec['item']} ({priority_marker} Priority)")
            recs_text.append(f"  {rec['recommendation']}")
        
        result = f"{response['title']}\n\n" + "\n".join(recs_text)
        
        if 'insights' in response and response['insights']:
            result += f"\n\nSeasonal Insights:\n" + "\n".join([f"- {insight}" for insight in response['insights']])
        
        if 'action_plan' in response and response['action_plan']:
            result += f"\n\nAction Plan:\n" + "\n".join([f"- {action}" for action in response['action_plan']])
        
        return result
    
    # Enhanced top products response
    elif response['type'] in ['top_products', 'top_products_enhanced']:
        if 'products' not in response:
            return "Error: No product data found. Please upload transaction data and run analysis first."
        
        products_text = []
        for i, product in enumerate(response['products'][:5], 1):
            products_text.append(f"{i}. {product['Item']} - {product['Count']} purchases ({product['Percentage']}%)")
        
        result = f"{response['title']}\n\n" + "\n".join(products_text)
        
        if 'insights' in response and response['insights']:
            result += f"\n\nProduct Insights:\n" + "\n".join([f"- {insight}" for insight in response['insights']])
        
        if 'category_performance' in response and response['category_performance']:
            result += f"\n\nCategory Performance:\n"
            for category, perf in response['category_performance'].items():
                result += f"- {category.title()}: {perf['count']} items, {perf['total_purchases']} purchases\n"
        
        if 'recommendations' in response and response['recommendations']:
            result += f"\n\nProduct Recommendations:\n" + "\n".join([f"- {rec}" for rec in response['recommendations']])
        
        return result
    
    # Enhanced metrics response
    elif response['type'] in ['metrics', 'metrics_enhanced']:
        if 'metrics' not in response:
            return "Error: No metrics data available. Please run the analysis first."
        
        metrics = response['metrics']
        metrics_text = [
            f"- Total Transactions: {metrics['total_transactions']:,}",
            f"- Unique Products: {metrics['total_items']:,}",
            f"- Average Items per Transaction: {metrics['avg_items_per_transaction']:.1f}"
        ]
        
        if 'date_range' in metrics:
            metrics_text.append(f"- Date Range: {metrics['date_range']['start']} to {metrics['date_range']['end']}")
        
        result = f"{response['title']}\n\n" + "\n".join(metrics_text)
        
        if 'performance_analysis' in response and response['performance_analysis']:
            result += f"\n\nPerformance Analysis:\n"
            for metric, status in response['performance_analysis'].items():
                result += f"- {metric.replace('_', ' ').title()}: {status}\n"
        
        if 'benchmarks' in response and response['benchmarks']:
            result += f"\n\nIndustry Benchmarks:\n"
            for benchmark, data in response['benchmarks'].items():
                result += f"- {benchmark.replace('_', ' ').title()}: {data['assessment']} (Current: {data['current']}, Industry: {data['industry_average']})\n"
        
        return result
    
    # Enhanced definition response
    elif response['type'] == 'definition':
        result = f"{response['title']}\n\n"
        result += f"Definition: {response['definition']}\n\n"
        
        if 'examples' in response and response['examples']:
            result += f"Examples:\n"
            for example in response['examples']:
                result += f"- {example}\n"
        
        if 'practical_application' in response:
            result += f"\nPractical Application: {response['practical_application']}"
        
        return result
    
    # Enhanced how-to response
    elif response['type'] == 'how_to':
        result = f"{response['title']}\n\n"
        
        if 'steps' in response:
            result += f"Steps:\n"
            for step in response['steps']:
                result += f"{step}\n"
        
        if 'strategies' in response:
            result += f"\nStrategies:\n"
            for strategy in response['strategies']:
                result += f"- {strategy}\n"
        
        if 'benefits' in response:
            result += f"\nBenefits:\n"
            for benefit in response['benefits']:
                result += f"- {benefit}\n"
        
        if 'implementation' in response:
            result += f"\nImplementation: {response['implementation']}"
        
        return result
    
    # Enhanced comparison response
    elif response['type'] == 'comparison':
        result = f"{response['title']}\n\n"
        
        if 'lift' in response:
            result += f"Lift:\n- {response['lift']['definition']}\n- {response['lift']['interpretation']}\n- {response['lift']['use_case']}\n\n"
        
        if 'confidence' in response:
            result += f"Confidence:\n- {response['confidence']['definition']}\n- {response['confidence']['interpretation']}\n- {response['confidence']['use_case']}\n\n"
        
        if 'recommendation' in response:
            result += f"Recommendation: {response['recommendation']}"
        
        return result
    
    # Enhanced business definition response
    elif response['type'] == 'business_definition':
        result = f"{response['title']}\n\n"
        result += f"Definition: {response['definition']}\n\n"
        
        if 'examples' in response and response['examples']:
            result += f"Examples:\n"
            for example in response['examples']:
                result += f"- {example}\n"
        
        if 'practical_application' in response:
            result += f"\nPractical Application: {response['practical_application']}"
        
        return result
    
    # Enhanced help response
    elif response['type'] in ['help', 'help_enhanced']:
        result = f"{response['title']}\n\n"
        
        if 'capabilities' in response:
            result += f"My Capabilities:\n"
            for capability in response['capabilities']:
                result += f"{capability}\n"
        
        if 'example_questions' in response:
            result += f"\nExample Questions:\n"
            for question in response['example_questions']:
                result += f"- {question}\n"
        
        if 'getting_started' in response:
            result += f"\nGetting Started:\n"
            for step in response['getting_started']:
                result += f"{step}\n"
        
        if 'pro_tips' in response:
            result += f"\nPro Tips:\n"
            for tip in response['pro_tips']:
                result += f"- {tip}\n"
        
        return result
    
    else:
        return response.get('message', "I'm here to help! Try asking about product combinations, customer segments, or seasonal trends.")

def render_action_plan():
    st.header("📋 Staff Action Plan")
    st.markdown("Data-driven recommendations for store operations and marketing.")
    
    # Get action plan from market basket analyzer
    action_plan = st.session_state.market_basket_analyzer.generate_action_plan(10)
    
    if not action_plan:
        if st.session_state.market_basket_analyzer.rules is None:
            st.warning("No action plan available. Please run the analysis first.")
        else:
            st.warning(
                "Market basket analysis completed, but no association rules were found. "
                "Try lowering support or confidence and rerunning the analysis."
            )
            if st.session_state.uploaded_data is not None:
                if st.button("🔄 Rerun Market Basket Analysis", key="rerun_action_plan"):
                    with st.spinner("Re-running market basket analysis..."):
                        try:
                            # Use very lenient values for rerun to ensure results
                            rerun_support = 0.001
                            rerun_confidence = 0.1
                            
                            baskets = st.session_state.data_processor.get_transaction_baskets()
                            st.session_state.market_basket_analyzer.prepare_transactions(baskets)
                            st.session_state.market_basket_analyzer.find_frequent_itemsets(rerun_support)
                            st.session_state.market_basket_analyzer.generate_association_rules(rerun_confidence)
                            st.success("✅ Market basket analysis rerun completed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Could not rerun analysis: {str(e)}")
        return
    
    # Display action plan
    for i, action in enumerate(action_plan, 1):
        with st.expander(f"🎯 Action {i}: {action['directive']}", expanded=i <= 3):
            st.markdown(f"**Why:** {action['why']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", action['confidence'])
            with col2:
                st.metric("Support", action['support'])
            with col3:
                st.metric("Lift", f"{action['lift']:.2f}")
    
    # Export action plan
    if st.button("📥 Export Action Plan"):
        action_plan_df = pd.DataFrame(action_plan)
        csv = action_plan_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_customer_segments():
    st.header("👥 Customer Segmentation")
    st.markdown("Understanding your customer base through data-driven segmentation.")
    
    # Get cluster summary
    cluster_summary = st.session_state.customer_segmentation.get_cluster_summary()
    
    if cluster_summary.empty:
        st.warning("No customer segmentation available. Please run the analysis first.")
        return
    
    # Display cluster summary
    st.subheader("📊 Segment Overview")
    st.dataframe(cluster_summary, use_container_width=True)
    
    # Create pie chart
    fig = st.session_state.visualizer.create_customer_segment_chart(cluster_summary)
    st.plotly_chart(fig, use_container_width=True, key='customer_segment_chart')
    
    # Get detailed analysis
    cluster_analysis = st.session_state.customer_segmentation.analyze_clusters(
        st.session_state.data_processor.data,
        st.session_state.data_processor.transaction_col,
        st.session_state.data_processor.item_col
    )
    
    # Display detailed analysis for each segment
    st.subheader("🔍 Segment Details")
    
    for cluster_id, analysis in cluster_analysis.items():
        with st.expander(f"📋 {analysis['segment_name']} (Cluster {cluster_id})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Customers", analysis['customer_count'])
                st.metric("Avg Items per Customer", f"{analysis['avg_items_per_customer']:.1f}")
            
            with col2:
                st.metric("Total Transactions", analysis['total_transactions'])
                st.metric("Unique Items", analysis['unique_items_purchased'])
            
            # Top items for this segment
            st.markdown("**Top Items:**")
            top_items_df = pd.DataFrame(list(analysis['top_items'].items()), 
                                      columns=['Item', 'Count']).head(5)
            st.dataframe(top_items_df, use_container_width=True)
    
    # PCA visualization
    if hasattr(st.session_state.customer_segmentation, 'pca_features'):
        st.subheader("🎯 Customer Segments Visualization")
        fig = st.session_state.visualizer.create_cluster_scatter_plot(
            st.session_state.customer_segmentation.pca_features,
            st.session_state.customer_segmentation.cluster_labels
        )
        st.plotly_chart(fig, use_container_width=True, key='cluster_scatter_plot')

def render_seasonal_trends():
    st.header("☀️ Seasonal Trend Analysis")
    st.markdown("Identify seasonal patterns and optimize your promotions.")
    
    # Check if seasonal analysis is available
    if st.session_state.seasonal_analyzer.seasonal_patterns is None:
        st.warning("No seasonal analysis available. Please make sure your data includes date information.")
        return
    
    # Season selector
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    selected_season = st.selectbox("Select Season:", seasons)
    
    # Get seasonal recommendations
    recommendations = st.session_state.seasonal_analyzer.get_seasonal_recommendations(selected_season, 10)
    
    if recommendations:
        st.subheader(f"🎯 {selected_season} Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = "🔴" if rec['priority'] == 'High' else "🟡"
            st.markdown(f"{priority_color} **{rec['item']}**")
            st.markdown(f"*{rec['recommendation']}*")
            st.markdown(f"*Seasonal Share: {rec['season_percentage']:.1f}% | Seasonality Index: {rec['seasonality_index']:.1f}*")
            st.markdown("---")
    
    # Seasonal comparison chart
    st.subheader("📊 Seasonal Performance Comparison")
    seasonal_comparison = st.session_state.seasonal_analyzer.get_seasonal_comparison()
    
    if not seasonal_comparison.empty:
        fig = st.session_state.visualizer.create_seasonal_heatmap(seasonal_comparison)
        st.plotly_chart(fig, use_container_width=True, key='seasonal_comparison_chart')
    
    # Top items for selected season
    st.subheader(f"🏆 Top Items in {selected_season}")
    top_seasonal_items = st.session_state.seasonal_analyzer.get_top_seasonal_items(selected_season, 10)
    
    if not top_seasonal_items.empty:
        fig = st.session_state.visualizer.create_top_sales_chart(top_seasonal_items)
        st.plotly_chart(fig, use_container_width=True, key='seasonal_top_items_chart')

def render_visualizations():
    st.header("📊 Interactive Visualizations")
    st.markdown("Explore your data through interactive charts and graphs.")
    
    # Get top items
    top_items = st.session_state.data_processor.get_top_items(15)
    
    if not top_items.empty:
        st.subheader("🏆 Top Selling Products")
        fig = st.session_state.visualizer.create_top_sales_chart(top_items)
        st.plotly_chart(fig, use_container_width=True, key='top_selling_products_chart')
    
    # Market basket visualization
    rules = st.session_state.market_basket_analyzer.rules
    
    if rules is not None and not rules.empty:
        st.subheader("🔗 Product Association Network")
        fig = st.session_state.visualizer.create_association_network(rules)
        st.plotly_chart(fig, use_container_width=True, key='association_network_chart')
        
        st.subheader("📈 Lift vs Confidence Analysis")
        fig = st.session_state.visualizer.create_lift_confidence_scatter(rules)
        st.plotly_chart(fig, use_container_width=True, key='lift_confidence_chart')
    
    # Seasonal visualization (if available)
    if st.session_state.seasonal_analyzer.seasonal_patterns is not None:
        st.subheader("☀️ Seasonal Trends")
        seasonal_comparison = st.session_state.seasonal_analyzer.get_seasonal_comparison()
        
        if not seasonal_comparison.empty:
            fig = st.session_state.visualizer.create_seasonal_heatmap(seasonal_comparison)
            st.plotly_chart(fig, use_container_width=True, key='seasonal_trends_heatmap')

def render_metrics():
    st.header("📈 Business Metrics Dashboard")
    st.markdown("Key performance indicators and summary statistics.")
    
    # Get summary statistics
    summary_stats = st.session_state.data_processor.get_summary_stats()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{summary_stats['total_transactions']:,}")
    
    with col2:
        st.metric("Unique Products", summary_stats['total_items'])
    
    with col3:
        st.metric("Avg Items/Transaction", f"{summary_stats['avg_items_per_transaction']:.1f}")
    
    with col4:
        st.metric("Total Records", f"{summary_stats['total_rows']:,}")
    
    # Market basket metrics
    mb_metrics = st.session_state.market_basket_analyzer.get_summary_metrics()
    
    st.subheader("🎯 Market Basket Analysis Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frequent Itemsets", mb_metrics['frequent_itemsets_count'])
    
    with col2:
        st.metric("Association Rules", mb_metrics['association_rules_count'])
    
    with col3:
        st.metric("Average Lift", f"{mb_metrics.get('avg_lift', 0):.2f}")
    
    with col4:
        st.metric("Average Confidence", f"{mb_metrics.get('avg_confidence', 0):.2f}")
    
    # Customer segmentation metrics
    if st.session_state.customer_segmentation.cluster_labels is not None:
        st.subheader("👥 Customer Segmentation Metrics")
        
        cluster_summary = st.session_state.customer_segmentation.get_cluster_summary()
        
        if not cluster_summary.empty:
            st.dataframe(cluster_summary, use_container_width=True)
    
    # Seasonal metrics (if available)
    if st.session_state.seasonal_analyzer.seasonal_patterns is not None:
        st.subheader("☀️ Seasonal Analysis Metrics")
        
        seasonal_comparison = st.session_state.seasonal_analyzer.get_seasonal_comparison()
        
        if not seasonal_comparison.empty:
            # Most seasonal items
            most_seasonal = seasonal_comparison.head(5)
            st.markdown("**Most Seasonal Items:**")
            st.dataframe(most_seasonal[['item', 'seasonality_index', 'peak_season']], use_container_width=True)
    
    # Create metrics dashboard
    all_metrics = {
        'total_transactions': summary_stats['total_transactions'],
        'total_items': summary_stats['total_items'],
        'association_rules_count': mb_metrics['association_rules_count'],
        'avg_lift': mb_metrics.get('avg_lift', 0)
    }
    
    fig = st.session_state.visualizer.create_metrics_dashboard(all_metrics)
    st.plotly_chart(fig, use_container_width=True, key='metrics_dashboard_chart')

if __name__ == "__main__":
    main()
