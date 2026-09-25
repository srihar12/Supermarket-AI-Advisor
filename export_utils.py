import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
import io

class ExportUtils:
    """Utilities for exporting analysis results to various formats."""
    
    @staticmethod
    def export_action_plan(action_plan: List[Dict], format_type: str = 'csv') -> str:
        """Export action plan to specified format."""
        if not action_plan:
            return ""
        
        df = pd.DataFrame(action_plan)
        
        if format_type.lower() == 'csv':
            return df.to_csv(index=False)
        elif format_type.lower() == 'json':
            return df.to_json(orient='records', indent=2)
        elif format_type.lower() == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Action Plan', index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    @staticmethod
    def export_association_rules(rules_df: pd.DataFrame, format_type: str = 'csv') -> str:
        """Export association rules to specified format."""
        if rules_df.empty:
            return ""
        
        # Select relevant columns for export
        export_columns = [
            'antecedents_str', 'consequents_str', 'support', 'confidence', 
            'lift', 'conviction', 'leverage'
        ]
        
        available_columns = [col for col in export_columns if col in rules_df.columns]
        export_df = rules_df[available_columns]
        
        if format_type.lower() == 'csv':
            return export_df.to_csv(index=False)
        elif format_type.lower() == 'json':
            return export_df.to_json(orient='records', indent=2)
        elif format_type.lower() == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Association Rules', index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    @staticmethod
    def export_customer_segments(cluster_analysis: Dict, format_type: str = 'csv') -> str:
        """Export customer segmentation results to specified format."""
        if not cluster_analysis:
            return ""
        
        # Convert to DataFrame
        segments_data = []
        for cluster_id, analysis in cluster_analysis.items():
            segments_data.append({
                'Cluster_ID': cluster_id,
                'Segment_Name': analysis['segment_name'],
                'Customer_Count': analysis['customer_count'],
                'Total_Transactions': analysis['total_transactions'],
                'Avg_Items_Per_Customer': analysis['avg_items_per_customer'],
                'Unique_Items_Purchased': analysis['unique_items_purchased'],
                'Avg_Transaction_Size': analysis['avg_transaction_size']
            })
        
        df = pd.DataFrame(segments_data)
        
        if format_type.lower() == 'csv':
            return df.to_csv(index=False)
        elif format_type.lower() == 'json':
            return df.to_json(orient='records', indent=2)
        elif format_type.lower() == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Customer Segments', index=False)
                
                # Add top items for each segment as separate sheets
                for cluster_id, analysis in cluster_analysis.items():
                    top_items_df = pd.DataFrame(list(analysis['top_items'].items()), 
                                              columns=['Item', 'Count'])
                    sheet_name = f'Cluster_{cluster_id}_Items'[:31]  # Excel sheet name limit
                    top_items_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    @staticmethod
    def export_seasonal_analysis(seasonal_patterns: Dict, format_type: str = 'csv') -> str:
        """Export seasonal analysis results to specified format."""
        if not seasonal_patterns:
            return ""
        
        # Convert to DataFrame
        seasonal_data = []
        for item, patterns in seasonal_patterns.items():
            row = {
                'Item': item,
                'Peak_Season': patterns['peak_season'],
                'Seasonality_Index': patterns['seasonality_index'],
                'Total_Purchases': patterns['total_purchases']
            }
            
            # Add seasonal percentages
            for season, percentage in patterns['seasonal_percentages'].items():
                row[f'{season}_Percentage'] = percentage
            
            seasonal_data.append(row)
        
        df = pd.DataFrame(seasonal_data)
        
        if format_type.lower() == 'csv':
            return df.to_csv(index=False)
        elif format_type.lower() == 'json':
            return df.to_json(orient='records', indent=2)
        elif format_type.lower() == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Seasonal Analysis', index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    @staticmethod
    def export_summary_report(summary_stats: Dict, mb_metrics: Dict, 
                            cluster_summary: Optional[pd.DataFrame] = None) -> str:
        """Generate a comprehensive summary report."""
        report = []
        
        # Title and timestamp
        report.append("# Supermarket AI Advisor - Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- **Total Transactions**: {summary_stats.get('total_transactions', 0):,}")
        report.append(f"- **Unique Products**: {summary_stats.get('total_items', 0):,}")
        report.append(f"- **Average Items per Transaction**: {summary_stats.get('avg_items_per_transaction', 0):.1f}")
        report.append(f"- **Association Rules Found**: {mb_metrics.get('association_rules_count', 0)}")
        report.append(f"- **Average Lift Score**: {mb_metrics.get('avg_lift', 0):.2f}")
        report.append("")
        
        # Market Basket Analysis Summary
        report.append("## Market Basket Analysis")
        report.append(f"- **Frequent Itemsets**: {mb_metrics.get('frequent_itemsets_count', 0)}")
        report.append(f"- **Association Rules**: {mb_metrics.get('association_rules_count', 0)}")
        report.append(f"- **Maximum Lift**: {mb_metrics.get('max_lift', 0):.2f}")
        report.append(f"- **Maximum Confidence**: {mb_metrics.get('max_confidence', 0):.2f}")
        report.append("")
        
        # Customer Segmentation Summary
        if cluster_summary is not None and not cluster_summary.empty:
            report.append("## Customer Segmentation")
            for _, row in cluster_summary.iterrows():
                report.append(f"- **Cluster {row['Cluster']}**: {row['Percentage']} of customers "
                            f"(Avg: {row['Avg_Items_Per_Customer']:.1f} items)")
            report.append("")
        
        # Key Insights
        report.append("## Key Insights")
        report.append("1. **Product Associations**: Strong product relationships identified through market basket analysis")
        report.append("2. **Customer Behavior**: Distinct customer segments with different purchasing patterns")
        report.append("3. **Seasonal Trends**: Products show varying popularity across seasons")
        report.append("4. **Business Opportunities**: Multiple opportunities for cross-promotion and optimization")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. **Store Layout**: Place frequently bought-together items in close proximity")
        report.append("2. **Promotions**: Create bundle deals based on strong product associations")
        report.append("3. **Marketing**: Target different customer segments with personalized offers")
        report.append("4. **Seasonal Planning**: Adjust inventory and promotions based on seasonal patterns")
        report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def create_download_filename(analysis_type: str, format_type: str) -> str:
        """Generate a filename for download."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = {
            'csv': 'csv',
            'json': 'json',
            'excel': 'xlsx',
            'markdown': 'md'
        }.get(format_type.lower(), 'txt')
        
        return f"supermarket_analysis_{analysis_type}_{timestamp}.{extension}"
