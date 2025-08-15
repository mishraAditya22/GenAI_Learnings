# LCEL ---> Langchain Experession Language
# This file is part of the Langchain Expression Language (LCEL) project.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from ll_config import AzureOpenAIConfig as LLMConfig
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("DOCUMENT_MODEL")
API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("OPENAI_API_BASE")
API_KEY = os.getenv("OPENAI_API_KEY")
if not MODEL_NAME or not API_VERSION or not AZURE_ENDPOINT or not API_KEY:
    print("API configuration missing !!")
    exit(1)

# Initialize the LLM with Azure OpenAI
try:
    llm = AzureChatOpenAI(
        azure_deployment=MODEL_NAME,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        temperature=0
    )
    print("✅ Successfully initialized Azure OpenAI")
except Exception as e:
    print(f"Error initializing Azure OpenAI: {e}")
    exit(1)

# #create the prompt template
# prompt = ChatPromptTemplate.from_template(
#     "You are an expert Gen AI leader and you help other with their queries about {topic}"
# )

# # Create the output parser
# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# # Invoke the chain with a sample input
# result = chain.invoke({"topic": "what is langchain chatPrompt template? & from_template method? explain me give basic examples."})
# print(result)


#Second example with retriver of qdrant vector store
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Initialize the embeddings
os.environ.pop("OPENAI_API_BASE", None)
embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),      # Azure deployment name
            azure_endpoint=os.getenv("EMBEDDING_OPENAI_API_BASE"),  # instead of openai_api_base
            openai_api_key=os.getenv("EMBEDDING_OPENAI_API_KEY"),
            openai_api_version=os.getenv("EMBEDDING_OPENAI_API_VERSION"),
            chunk_size=1
        )

# Initialize the Qdrant vector store
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="Employee_data",
    url="http://localhost:6333",
    content_payload_key="text"
)

#create the retriever
retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

user_question = "Model Context Protocol"

# Retrieve relevant context using the retriever
retrieved_docs = retriever.get_relevant_documents(user_question)
context = "\n".join([doc.page_content for doc in retrieved_docs])

print("Retrieved docs:", retrieved_docs)

# Use only ChatPromptTemplate for the chain
retriever_prompt = ChatPromptTemplate.from_template(
    "You are an expert Gen AI leader and you help other with their queries about {topic}. Here is some context of the knowledge you have, only answer by this: {context}, otherwsise say 'I don't know'."
)

# Create the output parser for the retriever
retriever_output_parser = StrOutputParser()

# Chain using ChatPromptTemplate
retrieval_chain = retriever_prompt | llm | retriever_output_parser

print("Retrieved Context:")
print(context)

# Pass both topic and context
result_retrieval = retrieval_chain.invoke({
    "topic": user_question,
    "context": context
})
print("Result from Retrieval Chain:")
print(result_retrieval)
