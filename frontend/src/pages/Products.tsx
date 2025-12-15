import { useState, useEffect } from 'react';
import { productApi, Product } from '../services/api';
import ProductCard from '../components/ProductCard';
import './Products.css';

const Products = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Search and filter states
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [minRating, setMinRating] = useState<number>(0);
  const [maxPrice, setMaxPrice] = useState<number>(0);

  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const cats = await productApi.getCategories();
        setCategories(cats);
      } catch (err) {
        console.error('Error fetching categories:', err);
      }
    };
    fetchCategories();
  }, []);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setLoading(true);
        let fetchedProducts: Product[];

        if (searchQuery.trim().length >= 2) {
          fetchedProducts = await productApi.searchProducts(searchQuery, {
            category: selectedCategory || undefined,
            min_rating: minRating > 0 ? minRating : undefined,
            max_price: maxPrice > 0 ? maxPrice : undefined,
            limit: 50,
          });
        } else {
          // If no search query, show top-rated products
          fetchedProducts = await productApi.getTopRatedProducts(
            minRating > 0 ? minRating : 4.0,
            selectedCategory || undefined,
            50
          );
        }

        setProducts(fetchedProducts);
        setError(null);
      } catch (err) {
        setError('Failed to load products');
        console.error('Error fetching products:', err);
      } finally {
        setLoading(false);
      }
    };

    const timeoutId = setTimeout(() => {
      fetchProducts();
    }, searchQuery.trim().length >= 2 ? 500 : 0);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, selectedCategory, minRating, maxPrice]);

  return (
    <div className="products-page">
      <div className="container">
        <h1>Products</h1>

        <div className="filters-section">
          <div className="search-bar">
            <input
              type="text"
              placeholder="Search products..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>

          <div className="filters">
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="filter-select"
            >
              <option value="">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>

            <select
              value={minRating}
              onChange={(e) => setMinRating(parseFloat(e.target.value))}
              className="filter-select"
            >
              <option value="0">Any Rating</option>
              <option value="4.0">4.0+ Stars</option>
              <option value="4.5">4.5+ Stars</option>
            </select>

            <input
              type="number"
              placeholder="Max Price"
              value={maxPrice || ''}
              onChange={(e) => setMaxPrice(parseFloat(e.target.value) || 0)}
              className="filter-input"
              min="0"
            />
          </div>
        </div>

        {loading && <div className="loading">Loading products...</div>}
        {error && <div className="error">{error}</div>}
        
        {!loading && !error && (
          <>
            <div className="results-count">
              Found {products.length} product{products.length !== 1 ? 's' : ''}
            </div>
            <div className="products-grid">
              {products.map((product) => {
                const productId = product.Product_ID || product.parent_asin || '';
                return (
                  <ProductCard key={productId} product={product} />
                );
              })}
            </div>
            {products.length === 0 && (
              <div className="no-results">
                No products found. Try adjusting your filters.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Products;
