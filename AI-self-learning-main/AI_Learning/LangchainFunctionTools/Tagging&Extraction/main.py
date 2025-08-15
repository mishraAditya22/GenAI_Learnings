import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# Load environment variables from .env file
load_dotenv()

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

# Define Pydantic model for tagging
class Tagging(BaseModel):
    """Tag the piece of text with particular information."""
    sentiment: str = Field(description="Sentiment of the text, e.g., positive, negative, neutral")
    language: str = Field(description="Language of the text (ISO 639-1 code, e.g., 'en' for English)")

# Convert the Pydantic model into OpenAI-compatible function spec
tagging_functions = [convert_to_openai_function(Tagging)]

# os.environ.pop("OPENAI_API_BASE", None)
# Connect to Azure OpenAI with correct parameter names
llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=azure_api_version,
    azure_endpoint=azure_api_base,
    api_key=azure_api_key,
    temperature=0
)

llm_with_function = llm.bind_tools(tagging_functions)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that tags text with sentiment and language."),
    ("user", "{input}"),
])

import json
# Create the full chain: prompt -> model -> parser
chain = prompt | llm_with_function | JsonOutputFunctionsParser()

# Invoke the chain with a test input
try:
    result = chain.invoke({
        "input": "I love earning money by being a software engineer."
    })
    # Print the parsed result
    print(result)
except Exception as e:
    print(f"Error: {e}\nTrying to print raw tool calls...")
    # Try to get raw tool calls if parsing fails
    raw_result = (prompt | llm_with_function).invoke({
        "input": "I love earning money by being a software engineer."
    })
    tool_calls = raw_result.additional_kwargs.get('tool_calls', [])
    for call in tool_calls:
        args = json.loads(call['function']['arguments'])
        print(f"Function: {call['function']['name']}, Sentiment: {args['sentiment']}, Language: {args['language']}")
