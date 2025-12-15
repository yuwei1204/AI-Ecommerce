#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
"""
Script to run the E-commerce RAG Chatbot application
"""

import uvicorn
import click
from pathlib import Path
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded environment variables from .env")
    else:
        logger.warning("No .env file found, using default settings")

def check_data_files():
    """Check if required data files exist"""
    base_dir = Path(__file__).parent.parent
    
    # Check for raw data files
    raw_dir = base_dir / 'data' / 'raw'
    raw_files = [
        raw_dir / 'Product_Information_Dataset.csv',
        raw_dir / 'Order_Data_Dataset.csv'
    ]
    
    # Check for processed data files
    processed_dir = base_dir / 'data' / 'processed'
    processed_files = [
        processed_dir / 'processed_products.csv',
        processed_dir / 'processed_orders.csv',
        processed_dir / 'product_embeddings.pkl'
    ]
    
    # Check if either raw or processed files exist
    raw_missing = [str(f) for f in raw_files if not f.exists()]
    processed_missing = [str(f) for f in processed_files if not f.exists()]
    
    if raw_missing and processed_missing:
        raise FileNotFoundError(
            f"Missing required data files. Need either:\n"
            f"Raw files: {', '.join(raw_missing)}\n"
            f"OR Processed files: {', '.join(processed_missing)}"
        )

@click.group()
def cli():
    """E-commerce RAG Chatbot CLI"""
    pass

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, help='Number of worker processes')
def api(host, port, reload, workers):
    """Run the API server"""
    try:
        # Setup
        setup_environment()
        check_data_files()
        
        # Get configuration from environment
        host = os.getenv('HOST', host)
        port = int(os.getenv('PORT', port))
        reload = os.getenv('RELOAD', '').lower() == 'true' or reload
        workers = int(os.getenv('WORKERS', workers))
        
        logger.info(f"Starting API server on {host}:{port}")
        
        # Run server
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers
        )
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

@cli.command()
@click.option('--input-file', required=True, help='Input file with queries')
@click.option('--output-file', required=True, help='Output file for responses')
def batch(input_file, output_file):
    """Run batch processing of queries"""
    try:
        from src.rag.assistant import ECommerceRAG
        from src.config import Settings
        import json
        
        # Setup
        setup_environment()
        check_data_files()
        
        # Initialize RAG system
        settings = Settings()
        assistant = ECommerceRAG(
            product_dataset_path=settings.PRODUCT_DATA_PATH,
            order_dataset_path=settings.ORDER_DATA_PATH
        )
        
        # Process queries
        with open(input_file, 'r') as f:
            queries = json.load(f)
        
        responses = []
        for query in queries:
            text = query.get('text', '')
            customer_id = query.get('customer_id')
            response = assistant.process_query(text, customer_id)
            responses.append({
                'query': query,
                'response': response
            })
        
        # Save responses
        with open(output_file, 'w') as f:
            json.dump(responses, f, indent=2)
        
        logger.info(f"Processed {len(queries)} queries")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise

@cli.command()
def chat():
    """Run interactive chat in the terminal"""
    try:
        # Local imports to keep startup light
        from src.rag.assistant import ECommerceRAG
        from src.config import Settings

        setup_environment()
        check_data_files()

        # Initialize RAG assistant
        settings = Settings()
        assistant = ECommerceRAG(
            product_dataset_path=settings.PRODUCT_DATA_PATH,
            order_dataset_path=settings.ORDER_DATA_PATH
        )

        customer_id = None
        print("Chat started. Type 'set customer <id>' to set a customer, or 'exit' to quit.")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            lower_input = user_input.lower()
            if lower_input in ("exit", "quit"):
                print("Goodbye!")
                break

            if lower_input.startswith("set customer"):
                parts = user_input.split()
                if len(parts) >= 3 and parts[2].isdigit():
                    customer_id = int(parts[2])
                    print(f"Customer set to {customer_id}")
                else:
                    print("Usage: set customer <customer_id>")
                continue

            # Handle regular query
            response = assistant.process_query(user_input, customer_id)
            print(f"Bot: {response}")

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise

if __name__ == "__main__":
    cli()