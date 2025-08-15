
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

class DocumentLoader:
    def __init__(self, pdf_path=None, web_url=None):
        self.pdf_path = pdf_path
        self.web_url = web_url

    def load_pdf(self):
        if not self.pdf_path:
            print("No PDF path provided.")
            return ""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        return "\n".join([page.page_content for page in pages])

    def load_web(self):
        if not self.web_url:
            print("No web URL provided.")
            return ""
        loader = WebBaseLoader(self.web_url)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])

    def load_all(self):
        print("Loading PDF and web content...")
        return self.load_pdf(), self.load_web()
