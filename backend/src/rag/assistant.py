import pandas as pd
import numpy as np
import re
import os
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECommerceRAG:
    def __init__(self, 
                 product_dataset_path: str, 
                 order_dataset_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG system"""
        self.product_df = pd.read_csv(product_dataset_path)
        self.order_df = pd.read_csv(order_dataset_path)
        
        # 設置本地模型路徑（backend/models/model_name）
        base_dir = Path(__file__).parent.parent.parent  # backend 目錄
        local_model_dir = base_dir / "models" / model_name
        
        # 如果本地模型不存在，則下載並保存
        if not local_model_dir.exists() or not any(local_model_dir.iterdir()):
            logger.info(f"Model not found locally. Downloading {model_name}...")
            # 先從 Hugging Face 下載到臨時位置
            temp_model = SentenceTransformer(model_name)
            # 保存到本地目錄
            local_model_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_model.save(str(local_model_dir))
            logger.info(f"Model saved to {local_model_dir}")
            self.model = temp_model
        else:
            logger.info(f"Loading model from local directory: {local_model_dir}")
            # 從本地目錄加載
            self.model = SentenceTransformer(str(local_model_dir))
        
        self._preprocess_data()
        self._create_product_embeddings()
    
    def _preprocess_data(self):
        """Preprocess datasets"""
        # Fill NaN values by dtype to avoid warnings
        for col in self.product_df.columns:
            if self.product_df[col].dtype == 'object':
                self.product_df[col] = self.product_df[col].fillna('')

        for col in self.order_df.columns:
            if self.order_df[col].dtype == 'object':
                self.order_df[col] = self.order_df[col].fillna('')

        # Handle both raw and processed product data formats
        if 'Product_Title' not in self.product_df.columns:
            # Raw format: map column names
            column_mapping = {
                'title': 'Product_Title',
                'average_rating': 'Rating',
                'description': 'Description',
                'features': 'Features',
                'price': 'Price',
                'parent_asin': 'Product_ID'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in self.product_df.columns:
                    self.product_df[new_col] = self.product_df[old_col]

        # Fill missing/empty description with features
        if 'Features' in self.product_df.columns:
            def fill_description(row):
                desc = str(row.get('Description', ''))
                # Check if description is empty, NaN, or empty list string
                if not desc or desc.lower() == 'nan' or desc.strip() == '[]':
                    features = str(row.get('Features', ''))
                    if features and features.lower() != 'nan' and features.strip() != '[]':
                        return features
                return desc
            
            self.product_df['Description'] = self.product_df.apply(fill_description, axis=1)

        # Handle both raw and processed order data formats
        if 'Order_DateTime' not in self.order_df.columns:
            # Raw format: combine Order_Date and Time
            if 'Order_Date' in self.order_df.columns and 'Time' in self.order_df.columns:
                self.order_df['Order_DateTime'] = pd.to_datetime(
                    self.order_df['Order_Date'].astype(str) + ' ' +
                    self.order_df['Time'].astype(str)
                )
            else:
                logger.warning("Order data missing datetime information")
                return
        else:
            # Processed format: just convert to datetime
            self.order_df['Order_DateTime'] = pd.to_datetime(self.order_df['Order_DateTime'])

        self.order_df = self.order_df.sort_values('Order_DateTime', ascending=False)
    
    def _create_product_embeddings(self):
        """Create product embeddings"""
        texts = self.product_df.apply(
            lambda x: f"{x['Product_Title']} {x['Description']}", 
            axis=1
        ).tolist()
        self.product_embeddings = self.model.encode(texts)
    
    def get_customer_orders(self, customer_id: int) -> List[Dict[str, Any]]:
        """Get orders for a specific customer"""
        customer_orders = self.order_df[self.order_df['Customer_Id'] == customer_id]
        return customer_orders.sort_values('Order_DateTime', ascending=False).to_dict('records')
    
    def get_high_priority_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get high priority orders"""
        high_priority = self.order_df[
            self.order_df['Order_Priority'].str.lower() == 'high'
        ]
        return high_priority.sort_values('Order_DateTime', ascending=False).head(limit).to_dict('records')
    
    def format_single_order(self, order: Dict[str, Any]) -> str:
        """Format single order details with HTML"""
        def escape_html(text: str) -> str:
            return (str(text)
                   .replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
        
        order_date = pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')
        product = escape_html(order['Product'])
        sales = float(order['Sales'])
        shipping = float(order['Shipping_Cost'])
        priority = escape_html(order['Order_Priority'])
        
        return (f"<p>Your order was placed on <strong>{order_date}</strong> "
                f"for <strong>{product}</strong>.</p>"
                f"<p>Total amount: <strong>${sales:.2f}</strong><br/>"
                f"Shipping cost: <strong>${shipping:.2f}</strong><br/>"
                f"Priority: <strong>{priority}</strong></p>")
    
    def format_high_priority_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Format high priority orders list with HTML"""
        if not orders:
            return "<p>No high priority orders found.</p>"
        
        def escape_html(text: str) -> str:
            return (str(text)
                   .replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
        
        response = f"<p><strong>Here are the {len(orders)} most recent high-priority orders:</strong></p><ul>"
        for i, order in enumerate(orders, 1):
            order_date = pd.Timestamp(order['Order_DateTime']).strftime('%Y-%m-%d %H:%M:%S')
            product = escape_html(order['Product'])
            sales = float(order['Sales'])
            shipping = float(order['Shipping_Cost'])
            customer_id = order['Customer_Id']
            
            response += (
                f"<li>On <strong>{order_date}</strong>, "
                f"<strong>{product}</strong> was ordered for <strong>${sales:.2f}</strong> "
                f"with a shipping cost of <strong>${shipping:.2f}</strong>. "
                f"(Customer ID: {customer_id})</li>"
            )
        response += "</ul>"
        return response
    
    def format_product_results(self, products: List[Dict[str, Any]]) -> str:
        """Format product results with HTML formatting for better display"""
        if not products:
            return "<p>No products found matching your criteria.</p>"
        
        response = "<p><strong>Here are some products that might interest you:</strong></p>"
        
        for i, product in enumerate(products, 1):
            # Extract title
            title = product.get('Product_Title', 'Unknown Product')
            
            # Extract and format rating
            rating = product.get('Rating', 0)
            try:
                rating = float(rating)
            except (ValueError, TypeError):
                rating = 0.0
            
            # Extract and format price
            price = product.get('Price', 0)
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = 0.0
            
            # Extract and format description
            description = product.get('Description', '')

            # Try to parse string representation of list
            if isinstance(description, str) and description.strip().startswith('[') and description.strip().endswith(']'):
                try:
                    parsed = ast.literal_eval(description)
                    if isinstance(parsed, (list, tuple)):
                        description = parsed
                except (ValueError, SyntaxError):
                    pass

            if isinstance(description, (list, tuple)):
                # If description is a list, join it and clean up
                description = ' '.join(str(d).strip() for d in description if d).strip()
            elif not isinstance(description, str):
                description = str(description) if description else ''
            
            # Clean up description
            if description:
                # Remove list brackets and quotes if present (e.g., "['text']" -> "text")
                # This is a fallback if ast.literal_eval failed or wasn't applicable but regex still needed
                if description.startswith('[') and description.endswith(']'):
                     description = re.sub(r"^\[['\"]?(.+?)['\"]?\]$", r"\1", description)
                description = re.sub(r"^['\"](.+?)['\"]$", r"\1", description)
                # Remove extra whitespace and limit length
                description = ' '.join(description.split())[:150]
            else:
                description = ''
            
            # Escape HTML special characters in text content
            def escape_html(text: str) -> str:
                return (str(text)
                       .replace('&', '&amp;')
                       .replace('<', '&lt;')
                       .replace('>', '&gt;')
                       .replace('"', '&quot;')
                       .replace("'", '&#39;'))
            
            # Format product entry with HTML
            response += '<div class="product-item">'
            response += f'<div class="product-title">{i}. {escape_html(title)}</div>'
            response += '<div class="product-details">'
            response += f'<div class="product-detail"><span class="icon">⭐</span> Rating: <strong>{rating:.1f} stars</strong></div>'
            response += f'<div class="product-detail"><span class="icon">💰</span> Price: <strong>${price:.2f}</strong></div>'
            if description:
                # Only add ellipsis if description was truncated
                desc_display = escape_html(description)
                if len(description) >= 150:
                    desc_display += '...'
                response += f'<div class="product-detail description-text"><span class="icon">📝</span> {desc_display}</div>'
            response += '</div>'
            response += '</div>'
        
        response += '<p><em>Let me know if you\'d like more details!</em></p>'
        return response
    
    def semantic_search(self, query: str, min_rating: Optional[float] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with rating and price filters
        """
        query_embedding = self.model.encode(query)
        similarities = np.dot(self.product_embeddings, query_embedding)
        
        # Create DataFrame with similarities
        results_df = self.product_df.copy()
        results_df['similarity'] = similarities
        
        # Apply rating filter if specified
        if min_rating is not None:
            results_df = results_df[results_df['Rating'] >= min_rating]
        
        # Apply price filter if specified
        if max_price is not None:
            results_df = results_df[results_df['Price'] <= max_price]
        
        # Sort by similarity and get top results
        results_df = results_df.sort_values('similarity', ascending=False).head(5)
        
        return results_df.to_dict('records')

    def process_query(self, query: str, customer_id: Optional[int] = None) -> str:
        """Process user query with improved filtering"""
        original_query = query
        query_lower = query.lower()
        
        # Extract rating requirement if present
        min_rating = None
        if 'above' in query_lower and any(char.isdigit() for char in query_lower):
            try:
                # Find rating value in query
                rating_idx = query_lower.find('above') + 5
                rating_str = ''.join(c for c in query_lower[rating_idx:] if c.isdigit() or c == '.')
                if rating_str:
                    min_rating = float(rating_str)
            except ValueError:
                pass
        
        # Extract price requirement if present (under, below, less than)
        max_price = None
        price_keywords = ['under', 'below', 'less than', 'cheaper than', 'up to']
        for keyword in price_keywords:
            if keyword in query_lower:
                try:
                    # Find price value after keyword
                    keyword_idx = query_lower.find(keyword) + len(keyword)
                    # Look for $ sign or number
                    price_part = query_lower[keyword_idx:keyword_idx+20]
                    # Extract number (with or without $)
                    price_match = re.search(r'\$?\s*(\d+(?:\.\d+)?)', price_part)
                    if price_match:
                        max_price = float(price_match.group(1))
                        break
                except (ValueError, AttributeError):
                    pass
        
        # Handle high priority orders query
        if 'high priority' in query_lower or ('recent' in query_lower and 'priority' in query_lower):
            # Extract limit from query (e.g., "fetch 20", "top 20", "20 most recent")
            limit = 10  # default
            limit_match = re.search(r'\b(\d+)\s*(?:most|recent|high|priority|orders?|top)?', query_lower)
            if limit_match:
                try:
                    extracted_limit = int(limit_match.group(1))
                    # Reasonable bounds check
                    if 1 <= extracted_limit <= 100:
                        limit = extracted_limit
                except ValueError:
                    pass
            
            orders = self.get_high_priority_orders(limit)
            return self.format_high_priority_orders(orders)
        
        # Handle regular order queries
        if any(keyword in query_lower for keyword in ['order', 'orders', 'purchase', 'bought']):
            if not customer_id:
                return "<p>Could you please provide your Customer ID?</p>"
            
            orders = self.get_customer_orders(customer_id)
            if not orders:
                return f"<p>No orders found for customer <strong>{customer_id}</strong></p>"
            return self.format_single_order(orders[0])
        
        # Handle product queries
        products = self.semantic_search(query, min_rating=min_rating, max_price=max_price)
        
        # Provide feedback if filters were applied but no results
        if not products:
            filter_msgs = []
            if min_rating is not None:
                filter_msgs.append(f"rating above {min_rating}")
            if max_price is not None:
                filter_msgs.append(f"price under ${max_price:.2f}")
            
            if filter_msgs:
                return f"<p>No products found matching your criteria (<strong>{', '.join(filter_msgs)}</strong>). Try adjusting your filters.</p>"
            else:
                return "<p>No products found matching your search. Try different keywords.</p>"
            
        return self.format_product_results(products)
