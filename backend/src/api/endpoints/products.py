from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from ...config import Settings

router = APIRouter()
settings = Settings()

# Load product data
PRODUCT_DF = pd.read_csv(settings.PRODUCT_DATA_PATH)
PRODUCT_DF.fillna('', inplace=True)

# Fix empty descriptions using feature_list or features
def fix_description(row):
    desc = str(row.get('Description', ''))
    if not desc or desc == 'nan' or desc.strip() == '[]':
        # Try feature_list first (processed data)
        if 'feature_list' in row and row['feature_list']:
             return str(row['feature_list'])
        # Try features (raw data)
        if 'features' in row and row['features']:
             return str(row['features'])
    return desc

PRODUCT_DF['Description'] = PRODUCT_DF.apply(fix_description, axis=1)

def find_product_by_id(product_id: str):
    """
    Find a product by ID, supporting both ASIN (string) and numeric ID formats.
    Returns the product row or None if not found.
    """
    # Try string match first (for ASIN format like "B09RCVDQ8M")
    string_match = PRODUCT_DF[PRODUCT_DF['Product_ID'].astype(str) == str(product_id)]
    if not string_match.empty:
        return string_match.iloc[0]
    
    # Try numeric match (if Product_ID is numeric)
    try:
        numeric_id = int(product_id)
        numeric_match = PRODUCT_DF[PRODUCT_DF['Product_ID'] == numeric_id]
        if not numeric_match.empty:
            return numeric_match.iloc[0]
    except (ValueError, TypeError):
        pass
    
    # Also check if there's a parent_asin column in raw data
    if 'parent_asin' in PRODUCT_DF.columns:
        asin_match = PRODUCT_DF[PRODUCT_DF['parent_asin'].astype(str) == str(product_id)]
        if not asin_match.empty:
            return asin_match.iloc[0]
    
    return None

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_products(
    query: str = Query(..., min_length=2),
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Search products with various filters
    """
    # Start with all products
    filtered_products = PRODUCT_DF.copy()
    
    # Apply search query across multiple fields
    if query:
        search_mask = (
            filtered_products['Product_Title'].str.contains(query, case=False, na=False) |
            filtered_products['Description'].str.contains(query, case=False, na=False) |
            filtered_products['Category'].str.contains(query, case=False, na=False)
        )
        filtered_products = filtered_products[search_mask]
    
    # Apply category filter
    if category:
        filtered_products = filtered_products[
            filtered_products['Category'].str.contains(category, case=False, na=False)
        ]
    
    # Apply rating filter
    if min_rating is not None:
        filtered_products = filtered_products[
            filtered_products['Rating'] >= min_rating
        ]
    
    # Apply price filter
    if max_price is not None:
        filtered_products = filtered_products[
            filtered_products['Price'] <= max_price
        ]
    
    if filtered_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No products found matching the criteria"
        )
    
    # Sort by relevance (currently using rating as a proxy)
    filtered_products = filtered_products.sort_values('Rating', ascending=False)
    
    # Limit results
    filtered_products = filtered_products.head(limit)
    
    return filtered_products.to_dict('records')

@router.get("/category/{category}", response_model=List[Dict[str, Any]])
async def get_products_by_category(
    category: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_rating: Optional[float] = None
):
    """
    Retrieve products in a specific category
    """
    # Filter by category
    category_products = PRODUCT_DF[
        PRODUCT_DF['Category'].str.contains(category, case=False, na=False)
    ].copy()
    
    if category_products.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No products found in category '{category}'"
        )
    
    # Apply rating filter if specified
    if min_rating is not None:
        category_products = category_products[
            category_products['Rating'] >= min_rating
        ]
    
    # Sort by rating and limit results
    category_products = category_products.sort_values('Rating', ascending=False)
    category_products = category_products.head(limit)
    
    return category_products.to_dict('records')

@router.get("/top-rated", response_model=List[Dict[str, Any]])
async def get_top_rated_products(
    min_rating: float = Query(4.0, ge=0, le=5),
    category: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get top-rated products with optional category filter
    """
    # Filter by rating
    top_products = PRODUCT_DF[PRODUCT_DF['Rating'] >= min_rating].copy()
    
    # Apply category filter if specified
    if category:
        top_products = top_products[
            top_products['Category'].str.contains(category, case=False, na=False)
        ]
    
    if top_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No products found matching the criteria"
        )
    
    # Sort by rating and limit results
    top_products = top_products.sort_values('Rating', ascending=False)
    top_products = top_products.head(limit)
    
    return top_products.to_dict('records')

@router.get("/{product_id}", response_model=Dict[str, Any])
async def get_product_by_id(product_id: str):
    """
    Get a single product by ID, supporting both ASIN (string) and numeric ID formats
    """
    product = find_product_by_id(product_id)
    
    if product is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product with ID {product_id} not found"
        )
    
    return product.to_dict()

@router.get("/recommendations/{product_id}", response_model=List[Dict[str, Any]])
async def get_product_recommendations(
    product_id: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get product recommendations based on category and rating.
    Supports both ASIN (string) and numeric ID formats.
    """
    # Get the target product using the helper function
    target_product = find_product_by_id(product_id)
    
    if target_product is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product with ID {product_id} not found"
        )
    
    # Get the product ID for comparison (handle both string and numeric)
    target_product_id = target_product['Product_ID']
    
    # Find similar products in the same category, excluding the target product
    similar_products = PRODUCT_DF[
        (PRODUCT_DF['Category'] == target_product['Category']) &
        (PRODUCT_DF['Product_ID'].astype(str) != str(target_product_id))
    ].copy()
    
    if similar_products.empty:
        raise HTTPException(
            status_code=404,
            detail="No similar products found"
        )
    
    # Sort by rating and limit results
    similar_products = similar_products.sort_values('Rating', ascending=False)
    similar_products = similar_products.head(limit)
    
    return similar_products.to_dict('records')

@router.get("/categories/list", response_model=List[str])
async def get_categories():
    """
    Get all unique product categories
    """
    categories = PRODUCT_DF['Category'].dropna().unique().tolist()
    categories = [cat for cat in categories if cat]  # Remove empty strings
    categories.sort()
    return categories