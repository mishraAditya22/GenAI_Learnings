# LangchainFunctionTools

This folder contains advanced examples and utilities for working with LangChain, Azure OpenAI, and Qdrant vector stores. It is designed to help you experiment with LangChain's Expression Language (LCEL), tool calling, and integration with external vector databases for retrieval-augmented generation (RAG).

## Structure

- `LCEL/`
  - `main.py`: Example of using LangChain Expression Language (LCEL) with Azure OpenAI and Qdrant for context-based retrieval and response generation.
- `ToolCalling/`
  - `main.py`: Example for tool calling and function tool integration (customize as needed).
- `Tagging&Extraction/`
  - `main.py`: Example of function calling with Pydantic models and Azure OpenAI for tagging text with sentiment and language. Includes robust error handling for Azure's tool call responses.
- `MultipleChains/`
  - `main.py`: Async example for running multiple LLM chains in parallel using `asyncio` and LangChain. Summarizes text and runs follow-up instructions concurrently. See its README for details.
- `utils/`
  - `ll_config.py`: Configuration utilities for Azure OpenAI and LangChain.

## LCEL Example

The LCEL example demonstrates:
- Loading environment variables for Azure OpenAI and embedding models
- Initializing AzureChatOpenAI and AzureOpenAIEmbeddings
- Connecting to Qdrant vector store for document retrieval
- Using ChatPromptTemplate and StrOutputParser for prompt chaining
- Retrieving relevant context and generating responses

### Usage
1. Set up your `.env` file with required Azure and Qdrant variables.
2. Start Qdrant locally (default: `http://localhost:6333`).
3. Run the LCEL example:
   ```bash
   cd LCEL
   python main.py
   ```

## Tagging & Extraction Example

Demonstrates function calling with Pydantic models and Azure OpenAI. Tags text with sentiment and language, and handles Azure's tool call responses robustly.

### Usage
1. Set up your `.env` file with Azure OpenAI variables.
2. Run the example:
   ```bash
   cd Tagging&Extraction
   python main.py
   ```

## MultipleChains Example

Demonstrates running multiple LLM chains in parallel using `asyncio` and LangChain. Summarizes text and runs several follow-up instructions concurrently.

### Usage
1. Set up your `.env` file with Azure OpenAI variables.
2. Run the example:
   ```bash
   cd MultipleChains
   python main.py
   ```

## Prerequisites
- Python 3.12+
- Azure OpenAI API access
- Qdrant server (for LCEL vector storage)
- Required Python packages (see `requirements.txt` in parent folder)

## Example Output
### LCEL
```
✅ Successfully initialized Azure OpenAI
Retrieved docs: [Document(...)]
Retrieved Context:
...
Result from Retrieval Chain:
...
```
### Tagging & Extraction
```
Sentiment: positive, Language: en
```
### MultipleChains
```
🔹 Summary:
Artificial intelligence (AI) has made rapid progress ...

🔸 Output 1 for instruction: Generate three critical thinking questions.
1. ...

🔸 Output 2 for instruction: Write a tweet summarizing this summary.
AI is transforming industries, but ethical ...

🔸 Output 3 for instruction: Translate the summary into French.
L'intelligence artificielle ...

🔸 Output 4 for instruction: List 2 pros and 2 cons based on this summary.
Pros: ...
Cons: ...
```


## References
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

For more details, see the code in each subfolder and the main repository README.

---

For more details, see the code in each subfolder and the main repository README.
