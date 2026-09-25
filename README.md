# Supermarket AI Advisor 🤖🛒

A data-driven decision-support tool for retail managers and staff that transforms raw transaction data into actionable business intelligence using machine learning and data mining techniques.

## Features

- 🤖 **Interactive AI Chatbot** - Natural language interface for asking business questions
- 🎯 **Market Basket Analysis** - Identifies frequently bought together items with staff action plans
- 👥 **Customer Segmentation** - Groups shoppers into categories using K-Means clustering
- ☀️ **Seasonal Trend Analysis** - Identifies product popularity patterns by season
- 📊 **Dynamic Visualizations** - Interactive charts for data patterns

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Technical Stack

- **Language**: Python 3.x
- **Framework**: Streamlit
- **Data Handling**: Pandas & NumPy
- **AI/ML**: mlxtend, scikit-learn
- **Visualizations**: Plotly Express

## Data Format

Upload a CSV file with transaction data containing:
- Transaction ID column
- Item Name column
- Date column (optional for seasonal analysis)

## Core Algorithms

- **Market Basket Analysis**: Apriori algorithm with support/confidence metrics
- **Customer Segmentation**: K-Means clustering
- **Association Rules**: Lift, conviction, and confidence metrics

Developed with Gemini AI Technology
