import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { orderApi } from '../services/api';
import { useUser } from '../contexts/UserContext';
import { FaCalendar, FaDollarSign, FaBox, FaUser, FaExclamationCircle } from 'react-icons/fa';
import './Orders.css';

interface Order {
  Order_DateTime?: string;
  Order_Date?: string;
  Time?: string;
  Product?: string;
  Sales?: number;
  Shipping_Cost?: number;
  Order_Priority?: string;
  Customer_Id?: number;
  [key: string]: any;
}

const Orders = () => {
  const { userId } = useUser();
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOrders = async () => {
      if (!userId) {
        setOrders([]);
        setError(null);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        const fetchedOrders = await orderApi.getCustomerOrders(userId, 20);
        setOrders(fetchedOrders);
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch orders');
        setOrders([]);
      } finally {
        setLoading(false);
      }
    };

    fetchOrders();
  }, [userId]);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return 'N/A';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch {
      return dateStr;
    }
  };

  const formatDateTime = (dateStr?: string, timeStr?: string) => {
    if (dateStr && timeStr) {
      try {
        const date = new Date(`${dateStr} ${timeStr}`);
        return date.toLocaleString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
      } catch {
        return `${dateStr} ${timeStr}`;
      }
    }
    return formatDate(dateStr);
  };

  if (!userId) {
    return (
      <div className="orders-page">
        <div className="container">
          <h1>My Orders</h1>
          <div className="no-user-message">
            <FaExclamationCircle className="no-user-icon" />
            <h2>User ID Not Set</h2>
            <p>Please set your Customer ID using the user icon in the top right corner to view your orders.</p>
            <Link to="/" className="go-home-link">
              Go to Home
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="orders-page">
      <div className="container">
        <h1>My Orders</h1>
        {userId && (
          <p className="user-id-display">
            <FaUser /> Customer ID: {userId}
          </p>
        )}

        {loading && (
          <div className="loading-message">Loading orders...</div>
        )}

        {error && (
          <div className="error-message">
            <FaExclamationCircle /> {error}
          </div>
        )}

        {!loading && !error && orders.length === 0 && (
          <div className="no-orders">
            <FaBox className="no-orders-icon" />
            <h2>No orders found</h2>
            <p>No orders found for Customer ID: {userId}</p>
          </div>
        )}

        {!loading && orders.length > 0 && (
          <div className="orders-list">
            <h2>Order History ({orders.length} orders)</h2>
            {orders.map((order, index) => (
              <div key={index} className="order-card">
                <div className="order-header">
                  <div className="order-date">
                    <FaCalendar />
                    <span>
                      {order.Order_DateTime
                        ? formatDate(order.Order_DateTime)
                        : formatDateTime(order.Order_Date, order.Time)}
                    </span>
                  </div>
                  {order.Order_Priority && (
                    <span
                      className={`priority-badge ${
                        order.Order_Priority.toLowerCase() === 'high'
                          ? 'high-priority'
                          : ''
                      }`}
                    >
                      {order.Order_Priority}
                    </span>
                  )}
                </div>

                <div className="order-content">
                  <div className="order-product">
                    <FaBox />
                    <span>{order.Product || 'N/A'}</span>
                  </div>

                  <div className="order-details">
                    <div className="order-detail-item">
                      <FaDollarSign />
                      <span>Sales: ${order.Sales?.toFixed(2) || '0.00'}</span>
                    </div>
                    {order.Shipping_Cost !== undefined && (
                      <div className="order-detail-item">
                        <span>Shipping: ${order.Shipping_Cost.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Orders;
