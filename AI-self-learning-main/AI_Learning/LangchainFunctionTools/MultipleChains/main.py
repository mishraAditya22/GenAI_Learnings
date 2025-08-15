import asyncio
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


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

# === Define LLMs ===
llm1 = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=azure_api_version,
    azure_endpoint=azure_api_base,
    api_key=azure_api_key,
    temperature=0
)
#user different llm for followup
llm2 = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=azure_api_version,
    azure_endpoint=azure_api_base,
    api_key=azure_api_key,
    temperature=0
)

# === Prompts ===
summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in 3-5 sentences:\n\n{text}"
)

followup_prompt = PromptTemplate.from_template(
    "Given the following summary:\n\n{summary}\n\nInstruction: {instruction}\n\nAnswer:"
)

# === Chains ===
summary_chain = summary_prompt | llm1
followup_chain = followup_prompt | llm2

# === Main Async Logic ===
async def run_async_chains(user_text: str, instructions: list[str]):
    # Step 1: Summarize using ainvoke
    summary = await summary_chain.ainvoke({"text": user_text})
    print("🔹 Summary:\n", summary)

    # Step 2: Run multiple follow-up chains in parallel using asyncio.gather
    tasks = [
        followup_chain.ainvoke({"summary": summary, "instruction": instruction})
        for instruction in instructions
    ]
    results = await asyncio.gather(*tasks)

    # Display Results
    for i, output in enumerate(results):
        print(f"\n🔸 Output {i+1} for instruction: {instructions[i]}\n{output.content}")

# === Run It ===
if __name__ == "__main__":
    user_text = """
    Artificial intelligence (AI) has made rapid progress in recent years, impacting various industries from healthcare to finance.
    As companies adopt AI technologies, ethical concerns have emerged, including bias in decision-making, job displacement, and lack of transparency.
    Policymakers are increasingly under pressure to regulate the development and deployment of AI systems.
    Some experts advocate for international collaboration, while others emphasize local governance and accountability.
    The future of AI will depend on how well these challenges are addressed in the coming decade.
    """

    instructions = [
        "Generate three critical thinking questions.",
        "Write a tweet summarizing this summary.",
        "Translate the summary into French.",
        "List 2 pros and 2 cons based on this summary."
    ]

    asyncio.run(run_async_chains(user_text, instructions))
