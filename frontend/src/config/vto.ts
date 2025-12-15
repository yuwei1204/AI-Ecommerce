// Virtual Try-On Configuration
export const VTO_CONFIG = {
  // API Endpoint
  SPACE_ID: "franciszzj/Leffa",
  API_NAME: "/leffa_predict_vt",
  
  // Default Parameters (Optimized for speed/quota)
  PARAMS: {
    step: 30,              // Minimum allowed by API is 30
    scale: 2.5,            // Standard guidance scale
    seed: 42,              // Fixed seed for consistency
    vt_model_type: "viton_hd",
    vt_garment_type: "upper_body", // Default to upper body (t-shirt)
    vt_repaint: false,      // Boolean: false
    ref_acceleration: true // Boolean: true (Speed up)
  },

  // UI Strings
  STRINGS: {
    loading: "Generating virtual try-on...",
    success: "Try-on complete!",
    error: "Virtual try-on failed. Please try again."
  }
};

