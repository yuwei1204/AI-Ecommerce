import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Product {
  Product_ID?: string;
  Product_Title: string;
  Description: string;
  Category: string;
  Price: number;
  Rating: number;
  Rating_Count?: number;
  Store?: string;
  [key: string]: any;
}

export interface ChatQuery {
  query: string;
  customer_id?: number;
}

export interface ChatResponse {
  response: string;
}

export const productApi = {
  // Search products
  searchProducts: async (
    query: string,
    options?: {
      category?: string;
      min_rating?: number;
      max_price?: number;
      limit?: number;
    }
  ): Promise<Product[]> => {
    const params = new URLSearchParams({ query });
    if (options?.category) params.append('category', options.category);
    if (options?.min_rating) params.append('min_rating', options.min_rating.toString());
    if (options?.max_price) params.append('max_price', options.max_price.toString());
    if (options?.limit) params.append('limit', options.limit.toString());
    
    const response = await api.get<Product[]>(`/products/search?${params}`);
    return response.data;
  },

  // Get product by ID (supports both ASIN string and numeric ID)
  getProductById: async (productId: string): Promise<Product> => {
    const response = await api.get<Product>(`/products/${productId}`);
    return response.data;
  },

  // Get top-rated products
  getTopRatedProducts: async (
    minRating: number = 4.0,
    category?: string,
    limit: number = 10
  ): Promise<Product[]> => {
    const params = new URLSearchParams({
      min_rating: minRating.toString(),
      limit: limit.toString(),
    });
    if (category) params.append('category', category);
    
    const response = await api.get<Product[]>(`/products/top-rated?${params}`);
    return response.data;
  },

  // Get products by category
  getProductsByCategory: async (
    category: string,
    limit: number = 10,
    minRating?: number
  ): Promise<Product[]> => {
    const params = new URLSearchParams({
      limit: limit.toString(),
    });
    if (minRating) params.append('min_rating', minRating.toString());
    
    const response = await api.get<Product[]>(`/products/category/${category}?${params}`);
    return response.data;
  },

  // Get product recommendations (supports both ASIN string and numeric ID)
  getProductRecommendations: async (
    productId: string,
    limit: number = 5
  ): Promise<Product[]> => {
    const response = await api.get<Product[]>(
      `/products/recommendations/${productId}?limit=${limit}`
    );
    return response.data;
  },

  // Get all categories
  getCategories: async (): Promise<string[]> => {
    const response = await api.get<string[]>(`/products/categories/list`);
    return response.data;
  },
};

export const chatApi = {
  // Send chat query
  sendQuery: async (query: ChatQuery): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/chat/query', query);
    return response.data;
  },
};

export const orderApi = {
  // Get customer orders
  getCustomerOrders: async (customerId: number, limit: number = 10): Promise<any[]> => {
    const response = await api.get<any[]>(`/orders/customer/${customerId}?limit=${limit}`);
    return response.data;
  },
};

export default api;
