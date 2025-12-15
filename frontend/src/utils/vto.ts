import { VTO_CONFIG } from "../config/vto";

// Declare global types for Gradio Client from CDN
declare global {
  interface Window {
    GradioClient: {
      Client: {
        connect: (spaceId: string, options?: { token?: string }) => Promise<GradioClientInstance>;
      };
      handle_file: (file: File) => any;
    };
  }
  
  interface GradioClientInstance {
    predict: (apiName: string, payload: any) => Promise<{ data: any[] }>;
  }
}

export interface VTOResult {
  imageUrl: string;
}

export async function performVirtualTryOn(
  userPhoto: File,
  hfToken?: string
): Promise<VTOResult> {
  try {
    // Check if Gradio Client is loaded
    if (!window.GradioClient) {
      throw new Error("Gradio Client not loaded. Please wait for the page to fully load.");
    }

    // 1. Get Product Image as Blob
    const productBlob = await getProductImageBlob();

    // 2. Connect to Gradio Client
    const token = hfToken?.trim() || undefined;
    if (token) console.log("Using provided HF Token...");
    
    const client = await window.GradioClient.Client.connect(VTO_CONFIG.SPACE_ID, { token });

    // 3. Prepare Payload
    // Convert Blob to File to avoid "Buffer is not defined" error
    const productFile = new File([productBlob], "product.png", { 
      type: productBlob.type 
    });

    // Note: Leffa expects:
    // src_image_path = Person (User Photo)
    // ref_image_path = Garment (Product Image)
    const handle_file = window.GradioClient.handle_file;
    const payload = {
      src_image_path: handle_file(userPhoto),
      ref_image_path: handle_file(productFile),
      ...VTO_CONFIG.PARAMS
    };

    // 4. Submit Job
    const result = await client.predict(VTO_CONFIG.API_NAME, payload);
    
    // 5. Handle Result
    // Result is typically an array: [Generated Image, Mask, DensePose]
    // We want the first one (Generated Image)
    if (result && result.data && result.data.length > 0) {
      const generatedImage = result.data[0];
      
      // Handle different return types (url string or object with url)
      let imageUrl: string | null = null;
      if (typeof generatedImage === "string") {
        imageUrl = generatedImage;
      } else if (generatedImage && typeof generatedImage === "object" && "url" in generatedImage) {
        imageUrl = (generatedImage as { url: string }).url;
      }

      if (imageUrl) {
        return { imageUrl };
      } else {
        throw new Error("Invalid result format from API");
      }
    } else {
      throw new Error("No image returned from API");
    }
  } catch (error) {
    console.error("VTO Error:", error);
    throw error;
  }
}

async function getProductImageBlob(): Promise<Blob> {
  // Use the product.png from public/assets
  const imgUrl = "/assets/product.png";
  const response = await fetch(imgUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch product image: ${response.statusText}`);
  }
  return await response.blob();
}

