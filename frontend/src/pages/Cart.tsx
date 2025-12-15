import { useCart } from '../hooks/useCart';
import { FaTrash, FaPlus, FaMinus, FaShoppingBag } from 'react-icons/fa';
import { Link } from 'react-router-dom';
import './Cart.css';

const Cart = () => {
  const { cartItems, removeFromCart, updateQuantity, getTotalPrice, clearCart } = useCart();

  if (cartItems.length === 0) {
    return (
      <div className="cart-page">
        <div className="container">
          <h1>Shopping Cart</h1>
          <div className="empty-cart">
            <FaShoppingBag className="empty-cart-icon" />
            <h2>Your cart is empty</h2>
            <p>Start shopping to add items to your cart!</p>
            <Link to="/products" className="shop-button">
              Browse Products
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="cart-page">
      <div className="container">
        <div className="cart-header">
          <h1>Shopping Cart</h1>
          <button onClick={clearCart} className="clear-cart-button">
            Clear Cart
          </button>
        </div>

        <div className="cart-content">
          <div className="cart-items">
            {cartItems.map((item) => {
              const productId = item.product.Product_ID || item.product.parent_asin || '';
              return (
                <div key={productId} className="cart-item">
                  <Link to={`/product/${productId}`} className="cart-item-image">
                    <img
                      src={`https://via.placeholder.com/150x150?text=${encodeURIComponent(
                        item.product.Product_Title?.substring(0, 10) || 'Product'
                      )}`}
                      alt={item.product.Product_Title}
                    />
                  </Link>

                  <div className="cart-item-info">
                    <Link to={`/product/${productId}`} className="cart-item-title">
                      {item.product.Product_Title}
                    </Link>
                    <div className="cart-item-category">{item.product.Category}</div>
                    <div className="cart-item-price">${item.product.Price?.toFixed(2)}</div>
                  </div>

                  <div className="cart-item-controls">
                    <div className="quantity-controls">
                      <button
                        onClick={() => updateQuantity(productId, item.quantity - 1)}
                        className="quantity-button"
                      >
                        <FaMinus />
                      </button>
                      <span className="quantity-value">{item.quantity}</span>
                      <button
                        onClick={() => updateQuantity(productId, item.quantity + 1)}
                        className="quantity-button"
                      >
                        <FaPlus />
                      </button>
                    </div>

                    <div className="cart-item-total">
                      ${((item.product.Price || 0) * item.quantity).toFixed(2)}
                    </div>

                    <button
                      onClick={() => removeFromCart(productId)}
                      className="remove-button"
                      aria-label="Remove item"
                    >
                      <FaTrash />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="cart-summary">
            <h2>Order Summary</h2>
            <div className="summary-row">
              <span>Subtotal:</span>
              <span>${getTotalPrice().toFixed(2)}</span>
            </div>
            <div className="summary-row">
              <span>Shipping:</span>
              <span>Free</span>
            </div>
            <div className="summary-row total">
              <span>Total:</span>
              <span>${getTotalPrice().toFixed(2)}</span>
            </div>
            <button className="checkout-button">Proceed to Checkout</button>
            <Link to="/products" className="continue-shopping">
              Continue Shopping
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Cart;

