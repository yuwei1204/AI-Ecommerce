from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ...rag.assistant import ECommerceRAG
from ...config import Settings
import logging

router = APIRouter()
settings = Settings()
logger = logging.getLogger(__name__)

# Initialize RAG assistant (singleton pattern)
_rag_assistant = None

def get_rag_assistant():
    """Get or initialize RAG assistant"""
    global _rag_assistant
    if _rag_assistant is None:
        try:
            _rag_assistant = ECommerceRAG(
                product_dataset_path=str(settings.PRODUCT_DATA_PATH),
                order_dataset_path=str(settings.ORDER_DATA_PATH),
                model_name=settings.EMBEDDING_MODEL
            )
            logger.info("RAG assistant initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG assistant: {str(e)}")
            raise
    return _rag_assistant

class ChatQuery(BaseModel):
    query: str
    customer_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str

@router.post("/query", response_model=ChatResponse)
async def chat_query(chat_query: ChatQuery):
    """
    Process a chat query using the RAG assistant
    """
    try:
        assistant = get_rag_assistant()
        response = assistant.process_query(
            query=chat_query.query,
            customer_id=chat_query.customer_id
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

