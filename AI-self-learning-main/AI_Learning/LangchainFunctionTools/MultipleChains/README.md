# MultipleChains: Async LLM Chaining with LangChain & Azure OpenAI

This example demonstrates how to run multiple language model chains in parallel using Python's `asyncio` and LangChain with Azure OpenAI.

## Features
- Summarizes a user-provided text using an LLM.
- Runs multiple follow-up instructions in parallel (e.g., generate questions, translate, tweet, pros/cons).
- Uses two separate Azure OpenAI LLM instances for summary and follow-up tasks.
- Async execution for fast, scalable multi-tasking.

## Requirements
- Python 3.8+
- Azure OpenAI access
- LangChain
- `python-dotenv` for environment variable management

## Setup
1. **Clone the repo or copy the folder.**
2. **Install dependencies:**
   ```bash
   pip install langchain langchain-openai python-dotenv
   ```
3. **Create a `.env` file in this directory with your Azure OpenAI credentials:**
   ```env
   OPENAI_API_KEY=your-azure-key-here
   OPENAI_API_BASE=https://your-endpoint.openai.azure.com/
   OPENAI_API_VERSION=yyyy-mm-dd
   DOCUMENT_MODEL=your-deployment-name
   ```

## Usage
Run the script:
```bash
python main.py
```

## How It Works
- The script loads Azure OpenAI credentials from `.env`.
- It defines two LLM chains:
  - `summary_chain`: Summarizes the input text.
  - `followup_chain`: Answers follow-up instructions based on the summary.
- The main async function:
  1. Gets a summary of the input text.
  2. Runs all follow-up instructions in parallel using `asyncio.gather`.
  3. Prints each result with its corresponding instruction.

## Example Output
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

## Customization
- Change the `user_text` and `instructions` in `main.py` to suit your use case.
- Add more instructions for additional parallel tasks.

## License
MIT
