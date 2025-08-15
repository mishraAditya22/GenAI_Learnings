from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class GetTextSplitters:
    def __init__(self, doc=None, chunk_size=26, chunk_overlap=4):
        self.doc = doc
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def splitByRecursiveCharacterTextSplitter(self):
        if not self.doc:
            print("No document provided.")
            return ""


        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n" 
        )
        return r_splitter.split_text(self.doc)

    def splitByCharacterTextSplitter(self):
        if not self.doc:
            print("No document provided.")
            return ""
        
        print("Chunk size:", self.chunk_size)

        print("Breaking text into chunks using CharacterTextSplitter")

        c_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        c_docs = c_splitter.split_text(self.doc)
        return c_docs
