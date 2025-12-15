import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';
import { FaShoppingCart, FaHome, FaBox, FaUser } from 'react-icons/fa';
import { useCart } from '../hooks/useCart';
import { useUser } from '../contexts/UserContext';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();
  const { cartItems } = useCart();
  const { userId, setUserId } = useUser();
  const [showUserDialog, setShowUserDialog] = useState(false);
  const [inputUserId, setInputUserId] = useState('');

  const cartItemCount = cartItems.reduce((sum, item) => sum + item.quantity, 0);

  const handleUserIconClick = () => {
    setShowUserDialog(true);
    setInputUserId(userId?.toString() || '');
  };

  const handleSetUserId = () => {
    const id = inputUserId.trim();
    if (id === '') {
      setUserId(null);
      setShowUserDialog(false);
      return;
    }
    
    const numId = parseInt(id, 10);
    if (isNaN(numId)) {
      alert('Please enter a valid number');
      return;
    }
    
    setUserId(numId);
    setShowUserDialog(false);
  };

  const handleClearUserId = () => {
    setUserId(null);
    setShowUserDialog(false);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <FaBox className="logo-icon" />
          <span>E-Commerce</span>
        </Link>
        
        <ul className="navbar-menu">
          <li>
            <Link 
              to="/" 
              className={location.pathname === '/' ? 'active' : ''}
            >
              <FaHome /> Home
            </Link>
          </li>
          <li>
            <Link 
              to="/products" 
              className={location.pathname === '/products' ? 'active' : ''}
            >
              Products
            </Link>
          </li>
          <li>
            <Link 
              to="/orders" 
              className={location.pathname === '/orders' ? 'active' : ''}
            >
              My Orders
            </Link>
          </li>
          <li>
            <Link 
              to="/cart" 
              className={`cart-link ${location.pathname === '/cart' ? 'active' : ''}`}
            >
              <FaShoppingCart />
              Cart
              {cartItemCount > 0 && (
                <span className="cart-badge">{cartItemCount}</span>
              )}
            </Link>
          </li>
          <li className="user-icon-container">
            <button 
              className="user-icon-button"
              onClick={handleUserIconClick}
              title={userId ? `User ID: ${userId}` : 'Set User ID'}
            >
              <FaUser />
              {userId && <span className="user-id-badge">{userId}</span>}
            </button>
          </li>
        </ul>

        {showUserDialog && (
          <div className="user-dialog-overlay" onClick={() => setShowUserDialog(false)}>
            <div className="user-dialog" onClick={(e) => e.stopPropagation()}>
              <h3>Set User ID</h3>
              <p>Enter your Customer ID to view orders and personalized content</p>
              <input
                type="text"
                placeholder="Enter Customer ID"
                value={inputUserId}
                onChange={(e) => setInputUserId(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSetUserId();
                  }
                }}
                autoFocus
              />
              <div className="user-dialog-actions">
                <button onClick={handleSetUserId} className="btn-primary">
                  Set ID
                </button>
                {userId && (
                  <button onClick={handleClearUserId} className="btn-secondary">
                    Clear
                  </button>
                )}
                <button onClick={() => setShowUserDialog(false)} className="btn-secondary">
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
