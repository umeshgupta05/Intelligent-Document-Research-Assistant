import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
}


class HybridRAG:
    def __init__(self, k=4):
        self.k = k
        self.embeddings = BedrockEmbeddings(
            model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            region_name=os.getenv("AWS_REGION"),
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        self.reset()

    def reset(self):
        self.chunks = []
        self.files = set()
        self.vector = None
        self.bm25 = None

    def add_files(self, paths):
        new_chunks = []
        for path in paths:
            if path.name in self.files:
                continue
            docs = LOADERS[path.suffix.lower()](str(path)).load()
            for doc in docs:
                doc.metadata["source"] = path.name
            new_chunks += self.splitter.split_documents(docs)
            self.files.add(path.name)
        if not new_chunks:
            return 0
        if self.vector is None:
            self.vector = FAISS.from_documents(new_chunks, self.embeddings)
        else:
            self.vector.add_documents(new_chunks)
        self.chunks += new_chunks
        self.bm25 = BM25Retriever.from_documents(self.chunks, k=self.k * 3)
        return len(new_chunks)

    def search(self, query):
        dense = self.vector.similarity_search(query, k=self.k * 3)
        sparse = self.bm25.invoke(query)
        scores = {}
        docs = {}
        for results in (dense, sparse):
            for rank, doc in enumerate(results):
                text = doc.page_content
                docs[text] = doc
                scores[text] = scores.get(text, 0) + 1 / (60 + rank)
        best = sorted(scores, key=scores.get, reverse=True)[: self.k]
        return [docs[text] for text in best]
