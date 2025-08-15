import sys
import os
# Add the root directory to the path (which contains llm_config.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils.llm_config import AzureOpenAIConfig

class LLMQueryService:
    def __init__(self):
        self.config = AzureOpenAIConfig()
        self.client = self.config.get_client()
        self.deployment_id = self.config.deployment_name

    def query_llm(self, user_query, search_results, max_tokens=1000):
        top_context = "\n".join([result.payload['text'] for result in search_results])
        llm_prompt = f"User question: {user_query}\nRelevant context:\n{top_context}\nAnswer the user's question using the context above."
        response = self.client.chat.completions.create(
            model=self.deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant, only answer the question using the context provided."},
                {"role": "user", "content": llm_prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
