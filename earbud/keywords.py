from sentence_transformers import SentenceTransformer
from typing import Tuple, List

from langchain import FAISS
#TODO change embeddings to fastest open source
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import Wikipedia

class KeywordSearcher:
    """Searches for keywords in a sentence using semantic search"""
    def __init__(self, keywords):
        self.keywords = keywords
 #       self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  #      self.keyword_embeddings = self.model.encode(keywords)
        self.doc_store = Wikipedia()
 #       self.vector_store = FAISS.from_texts("", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
    
    def search_docs(self, sentence):
        #TODO: add better similarity docstore search
        #TODO: add error handling
        #TODO: doc.metadata.page
        return [doc.summary for doc in self.doc_store.search(sentence)]

    def search_vector(self, sentence) -> List[str]:
        hits = self.vector_store.max_marginal_relevance_search(sentence)
        return hits
    
    def check_keywords(self, text, keywords) -> Tuple[bool, str]:
        """Naive keyword search"""
        for keyword in keywords:
            if keyword in text:
                return True, keyword
        return False, None


