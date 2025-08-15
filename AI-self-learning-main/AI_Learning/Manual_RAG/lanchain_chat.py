
import os
from langchain_openai import AzureChatOpenAI
from Loader.main import QdrantClientManager
from Embedding.main import EmbeddingManager
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print(f".env file not found at {env_path}")

# Check for Azure OpenAI settings
azure_api_key = os.getenv("OPENAI_API_KEY")
azure_api_base = os.getenv("OPENAI_API_BASE")
azure_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("DOCUMENT_MODEL")

if not all([azure_api_key, azure_api_base, azure_api_version, deployment_name]):
    raise ValueError(
        "ERROR: Azure OpenAI environment variables are not set correctly!\n"
        "Please ensure your .env file has all required Azure settings:\n"
        "OPENAI_API_KEY=your-azure-key-here\n"
        "OPENAI_API_BASE=https://your-endpoint.openai.azure.com/\n"
        "OPENAI_API_VERSION=yyyy-mm-dd\n"
        "DOCUMENT_MODEL=your-deployment-name\n"
    )

print(f"Using Azure OpenAI with deployment: {deployment_name}")
print(f"Azure endpoint: {azure_api_base}")

# Initialize the LLM with Azure OpenAI
try:
    # Print debug information
    print("Debug - Azure OpenAI configuration:")
    print(f"API Key: {'*****' + azure_api_key[-5:] if azure_api_key else 'None'}")
    print(f"API Base: {azure_api_base}")
    print(f"API Version: {azure_api_version}")
    print(f"Deployment Name: {deployment_name}")
    
    # Connect to Azure OpenAI with correct parameter names
    llm = AzureChatOpenAI(
        azure_deployment=deployment_name,
        api_version=azure_api_version,
        azure_endpoint=azure_api_base,
        api_key=azure_api_key,
        temperature=0
    )
    print("✅ Successfully initialized Azure OpenAI")
except Exception as e:
    print(f"Error initializing Azure OpenAI: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    # Fall back to a dummy LLM for testing if you have no API access
    try:
        # Try the new import path
        from langchain_community.llms import FakeListLLM
    except ImportError:
        # Fall back to old import path if needed
        from langchain.llms import FakeListLLM
    
    responses = [
        "This is a test response from a dummy LLM. The real model couldn't be loaded due to configuration issues.",
        "I found information about MCP servers. They are used for Model Context Protocol implementations.",
        "Based on the context, I can't answer that question. Thanks for asking!"
    ]
    llm = FakeListLLM(responses=responses)
    print("⚠️ Using FakeListLLM for testing only - your API key or model configuration has issues.")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Setup Qdrant vector DB and embedding manager
# Initialize the Qdrant client
qdrant_client = QdrantClientManager.initialize_client()
embedding_manager = EmbeddingManager()

# Check available collections in Qdrant
available_collections = [c.name for c in qdrant_client.get_collections().collections]
print(f"Available collections: {available_collections}")

# Use Employee_data collection if it exists, otherwise use the first available collection
collection_name = "Employee_data"
if collection_name not in available_collections and available_collections:
    collection_name = available_collections[0]
    print(f"Collection 'Employee_data' not found. Using '{collection_name}' instead.")
elif not available_collections:
    raise ValueError("No collections found in Qdrant. Please create and populate a collection first.")
else:
    print(f"Using collection: {collection_name}")

# Create a Qdrant vector store using the client
# Specify the content_payload_key to match your data structure
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_manager.embeddings,
    content_payload_key="text"  # This tells LangChain where to find the document content
)

# Setup conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the QA chain with memory
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

def chat_loop():
    print("Welcome to the RAG-powered chat system! Type 'quit' to exit.")
    print("-----------------------------------------------------------")
    
    while True:
        user_question = input("\nYour question: ")
        
        if user_question.lower() in ["quit", "exit", "q"]:
            print("Thank you for using the chat system. Goodbye!")
            break
        
        if not user_question.strip():
            print("Please enter a valid question.")
            continue
        
        try:
            # Debug: Print embedding information
            print(f"Generating embedding for query: {user_question}")
            
            # Run the query through the QA chain
            result = qa_chain({"query": user_question})
            answer = result["result"]
            
            print("\nAI: " + answer)
            
            # Optional: Print sources if available
            if result.get("source_documents"):
                print("\nSources:")
                sources = set()
                for i, doc in enumerate(result["source_documents"][:3]):  # Show up to 3 sources
                    print(f"\nSource {i+1}:")
                    # Print a snippet of the content
                    content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    print(f"Content: {content}")
                    
                    # Print metadata if available
                    if hasattr(doc, "metadata"):
                        print(f"Metadata: {doc.metadata}")
                    
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                
                if sources:
                    print("\nSource files:")
                    for i, source in enumerate(sources, 1):
                        print(f"{i}. {source}")
            else:
                print("\nNo source documents were returned. The RAG system may not have relevant information.")
                    
        except Exception as e:
            import traceback
            print(f"\nError: {str(e)}")
            print("Full error traceback:")
            traceback.print_exc()
            print("\nSorry, I couldn't process your question. Please try again.")

def verify_data_retrieval():
    """Check if we can retrieve data from the vector store"""
    print("\n--- Testing Vector Store Retrieval ---")
    
    # First, try a direct query to Qdrant to verify data exists
    try:
        results = qdrant_client.scroll(collection_name=collection_name, limit=1)
        points, next_offset = results
        if points:
            point = points[0]
            print("✅ Data exists in Qdrant collection!")
            if point.payload and 'text' in point.payload:
                print(f"Sample text: {point.payload['text'][:100]}...")
            print(f"Payload structure: {list(point.payload.keys()) if point.payload else None}")
        else:
            print("❌ No points found in the Qdrant collection.")
            return False
    except Exception as e:
        print(f"❌ Error accessing Qdrant directly: {str(e)}")
        return False
    
    # Now try retrieving through the LangChain vector store
    print("\n--- Testing LangChain Vector Store Integration ---")
    try:
        # Create a test query vector
        query = "test query"
        query_vector = embedding_manager.embed_query(query)
        
        # Try to search using the vector store
        docs = vector_store.similarity_search(query, k=1)
        
        if docs:
            print("✅ Successfully retrieved a document via LangChain!")
            print(f"Document content: {docs[0].page_content[:100]}...")
            return True
        else:
            print("❌ No documents returned from LangChain vector store.")
            return False
    except Exception as e:
        print(f"❌ Error with LangChain vector store: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Run the chat loop if this script is executed directly
if __name__ == "__main__":
    # First verify data can be retrieved
    data_available = verify_data_retrieval()
    if data_available:
        print("\nStarting chat interface...\n")
        chat_loop()
    else:
        print("\nWARNING: Could not retrieve data from the vector store.")
        user_choice = input("Do you want to continue with the chat interface anyway? (y/n): ")
        if user_choice.lower() in ["y", "yes"]:
            print("\nStarting chat interface...\n")
            chat_loop()
        else:
            print("Exiting. Please check your data and try again.")