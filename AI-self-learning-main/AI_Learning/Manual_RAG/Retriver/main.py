from Langchain.ChatWithData.llm_config import AzureOpenAIConfig

# Extract top relevant document(s) from search_results
top_context = "\n".join([result.payload['text'] for result in search_results])

# Prepare prompt for LLM
llm_prompt = f"User question: {query}\nRelevant context:\n{top_context}\nAnswer the user's question using the context above."

# Initialize Azure OpenAI client
config = AzureOpenAIConfig()
client = config.get_client()

# Query the LLM (assumes chat/completions API)
response = client.chat.completions.create(
    model=config.deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": llm_prompt}
    ],
    max_tokens=512
)

print("LLM Response:")
print(response.choices[0].message.content)