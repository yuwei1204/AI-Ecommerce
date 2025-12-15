import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface UserContextType {
  userId: number | null;
  setUserId: (id: number | null) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const USER_ID_STORAGE_KEY = 'ecommerce_user_id';

export const UserProvider = ({ children }: { children: ReactNode }) => {
  const [userId, setUserIdState] = useState<number | null>(() => {
    // Initialize from localStorage
    const stored = localStorage.getItem(USER_ID_STORAGE_KEY);
    if (stored) {
      const parsed = parseInt(stored, 10);
      return isNaN(parsed) ? null : parsed;
    }
    return null;
  });

  const setUserId = (id: number | null) => {
    setUserIdState(id);
    if (id !== null) {
      localStorage.setItem(USER_ID_STORAGE_KEY, id.toString());
    } else {
      localStorage.removeItem(USER_ID_STORAGE_KEY);
    }
  };

  return (
    <UserContext.Provider value={{ userId, setUserId }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

