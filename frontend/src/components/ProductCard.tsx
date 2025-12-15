import { Link } from 'react-router-dom';
import { FaStar } from 'react-icons/fa';
import { Product } from '../services/api';
import './ProductCard.css';

interface ProductCardProps {
  product: Product;
}

const ProductCard = ({ product }: ProductCardProps) => {
  const productId = product.Product_ID || product.parent_asin || '';
  const title = product.Product_Title || '';
  const price = product.Price || 0;
  const rating = product.Rating || 0;
  const ratingCount = product.Rating_Count || 0;
  const category = product.Category || '';

  // Generate placeholder image URL based on product title
  const imageUrl = `https://placehold.co/300x300?text=${encodeURIComponent(title.substring(0, 20))}`;

  return (
    <Link to={`/product/${productId}`} className="product-card">
      <div className="product-image">
        <img src={imageUrl} alt={title} />
      </div>
      <div className="product-info">
        <h3 className="product-title">{title}</h3>
        <div className="product-category">{category}</div>
        <div className="product-rating">
          <FaStar className="star-icon" />
          <span>{rating.toFixed(1)}</span>
          {ratingCount > 0 && (
            <span className="rating-count">({ratingCount.toLocaleString()})</span>
          )}
        </div>
        <div className="product-price">${price.toFixed(2)}</div>
      </div>
    </Link>
  );
};

export default ProductCard;

