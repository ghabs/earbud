from sentence_transformers import SentenceTransformer
from typing import Tuple, List

from langchain import FAISS
#TODO change embeddings to fastest open source
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import Wikipedia

class EmbedSearch:
    """Searches for keywords in a sentence using semantic search"""
    def __init__(self, keywords):
        self.keywords = keywords
 #       self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
 #      self.keyword_embeddings = self.model.encode(keywords)
        """Context: Running list of sentences to search for keywords in"""
        self.context = []
        """Document store that can be customized and searched - defaults to wikipedia"""
        self.doc_store = Wikipedia()
        """Knowledge graph that can be customized and searched"""
 #       self.vector_store = FAISS.from_texts("", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
    
    def add_context(self, context):
        self.context.append(context)

    def search_docs(self, sentence):
        #TODO: add better similarity docstore search
        #TODO: add error handling
        #TODO: doc.metadata.page
        return [doc for doc in self.doc_store.search(sentence)]

    def search_vector(self, sentence) -> List[str]:
        hits = self.vector_store.max_marginal_relevance_search(sentence)
        return hits
    
    def check_keywords(self, text, keywords) -> Tuple[bool, str]:
        """Naive keyword search"""
        if len(keywords) == 0:
            return False, None
        for keyword in keywords:
            if keyword in text:
                return True, keyword
        return False, None


