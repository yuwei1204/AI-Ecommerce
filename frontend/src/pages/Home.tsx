import { useState, useEffect } from 'react';
import { productApi, Product } from '../services/api';
import ProductCard from '../components/ProductCard';
import './Home.css';

const Home = () => {
  const [topProducts, setTopProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTopProducts = async () => {
      try {
        setLoading(true);
        const products = await productApi.getTopRatedProducts(4.0, undefined, 12);
        setTopProducts(products);
        setError(null);
      } catch (err) {
        setError('Failed to load products');
        console.error('Error fetching top products:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchTopProducts();
  }, []);

  return (
    <div className="home">
      <div className="home-hero">
        <h1>Welcome to E-Commerce</h1>
        <p>Discover amazing products with AI-powered search</p>
      </div>

      <div className="container">
        <section className="featured-section">
          <h2>Top Rated Products</h2>
          {loading && <div className="loading">Loading products...</div>}
          {error && <div className="error">{error}</div>}
          {!loading && !error && (
            <div className="products-grid">
              {topProducts.map((product) => {
                const productId = product.Product_ID || product.parent_asin || '';
                return (
                  <ProductCard key={productId} product={product} />
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default Home;

