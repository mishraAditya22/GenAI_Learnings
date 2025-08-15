
import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

class AzureOpenAIConfig:
    """
    Encapsulates Azure OpenAI configuration and client creation.
    """
    def __init__(self):
        # Load .env from default location (See <attachments> above for file contents. You may not need to search or read the file again.)
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.endpoint = os.getenv("OPENAI_API_BASE")  # e.g. https://<resource>.openai.azure.com
        self.api_version = os.getenv("OPENAI_API_VERSION")   # e.g. 2023-05-15
        self.deployment_name = os.getenv("DOCUMENT_MODEL")   # Your deployment name

    def validate(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        if not self.endpoint:
            # See <attachments> above for file contents. You may not need to search or read the file again.
            raise ValueError("OPENAI_API_BASE not set in environment. (See <attachments> above for file contents. You may not need to search or read the file again.)")
        if not self.api_version:
            raise ValueError("OPENAI_API_VERSION not set in environment.")
        if not self.deployment_name:
            raise ValueError("DOCUMENT_MODEL not set in environment.")

    def get_client(self) -> AzureOpenAI:
        """Get configured Azure OpenAI client."""
        self.validate()
        # print(f"Using Azure OpenAI endpoint: {self.endpoint} with deployment: {self.deployment_name}")
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
