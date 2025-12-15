import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { productApi, Product } from '../services/api';
import { useCart } from '../hooks/useCart';
import { FaStar, FaShoppingCart, FaArrowLeft } from 'react-icons/fa';
import ProductCard from '../components/ProductCard';
import { performVirtualTryOn } from '../utils/vto';
import { VTO_CONFIG } from '../config/vto';
import './ProductDetail.css';

const ProductDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { addToCart } = useCart();
  
  const [product, setProduct] = useState<Product | null>(null);
  const [recommendations, setRecommendations] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [quantity, setQuantity] = useState(1);
  
  // VTO states
  const [vtoLoading, setVtoLoading] = useState(false);
  const [vtoError, setVtoError] = useState<string | null>(null);
  const [vtoResultUrl, setVtoResultUrl] = useState<string | null>(null);
  const [showVtoOverlay, setShowVtoOverlay] = useState(false);
  const [hfToken, setHfToken] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchProduct = async () => {
      if (!id) return;

      try {
        setLoading(true);
        setError(null);
        const fetchedProduct = await productApi.getProductById(id);
        setProduct(fetchedProduct);

        // Fetch recommendations
        try {
          const recs = await productApi.getProductRecommendations(id, 4);
          setRecommendations(recs);
        } catch (err) {
          console.error('Error fetching recommendations:', err);
        }
      } catch (err) {
        setError('Product not found');
        console.error('Error fetching product:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProduct();
  }, [id]);

  const handleAddToCart = () => {
    if (product) {
      addToCart(product, quantity);
      alert('Product added to cart!');
    }
  };

  const handleVtoUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const userFile = event.target.files?.[0];
    if (!userFile) return;

    setVtoLoading(true);
    setVtoError(null);
    setVtoResultUrl(null);

    try {
      const result = await performVirtualTryOn(userFile, hfToken || undefined);
      setVtoResultUrl(result.imageUrl);
      setShowVtoOverlay(true);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : VTO_CONFIG.STRINGS.error;
      setVtoError(errorMessage);
      alert(`${VTO_CONFIG.STRINGS.error}\n\nDetails: ${errorMessage}`);
    } finally {
      setVtoLoading(false);
      // Clear input so same file can be selected again if needed
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleCloseVto = () => {
    setShowVtoOverlay(false);
    setVtoResultUrl(null);
  };

  if (loading) {
    return (
      <div className="product-detail">
        <div className="container">
          <div className="loading">Loading product...</div>
        </div>
      </div>
    );
  }

  if (error || !product) {
    return (
      <div className="product-detail">
        <div className="container">
          <div className="error">{error || 'Product not found'}</div>
          <button onClick={() => navigate('/products')} className="back-button">
            <FaArrowLeft /> Back to Products
          </button>
        </div>
      </div>
    );
  }

  // Use fixed product.png for all products
  const imageUrl = '/assets/product.png';

  return (
    <div className="product-detail">
      <div className="container">
        <button onClick={() => navigate(-1)} className="back-button">
          <FaArrowLeft /> Back
        </button>

        <div className="product-detail-content">
          <div className="product-image-section">
            <div className="main-image-container">
              <img src={imageUrl} alt={product.Product_Title} crossOrigin="anonymous" />
              {showVtoOverlay && vtoResultUrl && (
                <div className="vto-result">
                  <img src={vtoResultUrl} alt="Virtual Try-On Result" />
                  <button className="close-btn" onClick={handleCloseVto}>&times;</button>
                </div>
              )}
            </div>
          </div>

          <div className="product-info-section">
            <h1>{product.Product_Title}</h1>
            
            <div className="product-meta">
              <div className="product-rating">
                <FaStar className="star-icon" />
                <span>{product.Rating?.toFixed(1) || 'N/A'}</span>
                {product.Rating_Count && (
                  <span className="rating-count">
                    ({product.Rating_Count.toLocaleString()} reviews)
                  </span>
                )}
              </div>
              <div className="product-category">{product.Category}</div>
            </div>

            <div className="product-price">${product.Price?.toFixed(2) || '0.00'}</div>

            <div className="product-description">
              <h3>Description</h3>
              <p className="description-text">
                {product.Description || 'No description available.'}
              </p>
            </div>

            <div className="product-actions">
              <div className="quantity-selector">
                <label>Quantity:</label>
                <input
                  type="number"
                  min="1"
                  value={quantity}
                  onChange={(e) => setQuantity(parseInt(e.target.value) || 1)}
                />
              </div>
              <button onClick={handleAddToCart} className="add-to-cart-button">
                <FaShoppingCart /> Add to Cart
              </button>
            </div>

            <hr className="divider" />

            {/* AI Virtual Try-On Section */}
            <div className="vto-section">
              <h2>
                <span className="ai-badge">AI</span> Virtual Try-On
              </h2>
              <p className="vto-desc">Upload your photo to see how this item looks on you.</p>
              
              <div className="vto-controls">
                {/* HF Token Input (Optional but recommended) */}
                <div className="token-input-group">
                  <label htmlFor="hf-token" className="sr-only">Hugging Face Token</label>
                  <input
                    type="password"
                    id="hf-token"
                    placeholder="Optional: Paste HF Token (hf_...)"
                    value={hfToken}
                    onChange={(e) => setHfToken(e.target.value)}
                  />
                </div>

                <div className="file-upload-wrapper">
                  <button
                    className="secondary-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={vtoLoading}
                  >
                    {vtoLoading ? 'Processing...' : 'Upload Your Photo'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    id="user-photo-input"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={handleVtoUpload}
                    disabled={vtoLoading}
                  />
                </div>
              </div>

              {/* VTO Status */}
              {vtoLoading && (
                <div className="vto-status">
                  <div className="spinner"></div>
                  <span>{VTO_CONFIG.STRINGS.loading}</span>
                </div>
              )}

              {vtoError && (
                <div className="vto-error">
                  <p>{vtoError}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {recommendations.length > 0 && (
          <section className="recommendations-section">
            <h2>You May Also Like</h2>
            <div className="recommendations-grid">
              {recommendations.map((rec) => {
                const recId = rec.Product_ID || rec.parent_asin || '';
                return <ProductCard key={recId} product={rec} />;
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

export default ProductDetail;
