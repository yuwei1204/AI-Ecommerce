# E-Commerce RAG Chatbot

這是一個基於檢索增強生成 (RAG) 的電子商務產品搜尋與訂單管理聊天機器人系統。該系統提供產品的語義搜尋功能，並透過對話介面處理訂單查詢。

專案包含兩個主要部分：
- **Backend**: 使用 Python 和 FastAPI 構建的後端服務，負責 RAG 邏輯、資料處理與 API。
- **Frontend**: 使用 React 和 Vite 構建的前端介面。

## 專案結構

```
ecommerce-rag-chatbot/
├── backend/            # 後端程式碼 (FastAPI, RAG)
├── frontend/           # 前端程式碼 (React, Vite)
└── README.md           # 專案說明文件
```

## 執行步驟

請依照以下步驟分別設定並啟動後端與前端服務。

### 1. 後端設定 (Backend)

**前置作業：**
- 確保已安裝 Python 3.8+

**步驟：**

1. 進入後端目錄：
   ```bash
   cd backend
   ```

2. 建立並啟動虛擬環境：
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. 安裝依賴套件：
   ```bash
   # Windows (推薦：自動偵測 GPU 並安裝對應 PyTorch)
   powershell -ExecutionPolicy ByPass -File setup.ps1

   # 手動安裝 (自行確認 PyTorch 是否符合要求)
   pip install -r requirements.txt
   ```

4. 設定環境變數：
   ```bash
   cp .env.example .env
   # 若有需要，請編輯 .env 檔案中的設定
   ```

5. 資料準備：
   - 請確保原始資料檔案 (`Product_Information_Dataset.csv` 和 `Order_Data_Dataset.csv`) 位於 `backend/data/raw/` 目錄中。
   - 執行資料預處理：
     ```bash
     python scripts/preprocess_data.py
     ```

6. 啟動後端伺服器：
   ```bash
   python scripts/run.py api
   ```
   - API 伺服器將在 http://localhost:8000 啟動。
   - 也可以透過命令行介面測試聊天功能：`python scripts/run.py chat`

---

### 2. 前端設定 (Frontend)

**前置作業：**
- 確保已安裝 Node.js (建議 v16+)

**步驟：**

1. 開啟一個新的終端機視窗，並進入前端目錄：
   ```bash
   cd frontend
   ```

2. 安裝依賴套件：
   ```bash
   npm install
   ```

3. 啟動開發伺服器：
   ```bash
   npm run dev
   ```
   - 前端應用程式通常會在 http://localhost:3000 啟動 (請依終端機顯示為準)。

## 功能說明

### 後端 API
後端提供以下主要的 API 端點 (可透過 http://localhost:8000/docs 查看完整文件)：

- **產品搜尋**: `GET /products/search`
- **訂單查詢**: `GET /orders/customer/{customer_id}`

### 聊天指令範例 (CLI 模式)
若使用 `python scripts/run.py chat` 進行測試：
- 設定客戶 ID: `set customer 37077`
- 詢問產品: `What are the top 5 highly-rated guitar products?`
- 詢問產品: `Show me microphones under $200`
- 詢問訂單: `What are the details of my last order?`
- 詢問高優先級訂單: `Fetch 5 most recent high-priority orders`


## 測試

若要執行後端測試：
```bash
cd backend
pytest tests/
```

