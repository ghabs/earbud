from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from typing import List
# Retrieve Concepts from Text and Search for Documents
# ====================================================
#
from dotenv import load_dotenv
load_dotenv()

#TODO: Change from default openai to a user defined one

class ConceptSearcher:
    """Search for documents related to a concept."""

    def __init__(self, llm=None, docstore=None, limit=1):
        if not llm:
            self.llm = OpenAI()
        else:
            self.llm = llm
        self.docstore = docstore
        self.context = []
        self.limit = limit

    def identify_concept(self, text: str) -> str:
        """Identify the concept in the text."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="From the following bullet points, identify 1 to 3 key concepts where more research could be useful:\n {text}")
        prompt = prompt.format(text=text)
        #TODO: add error handling, add config for llm
        response = self.llm(prompt)
        return response

    def search(self, text: str, k: int = 4) -> List[Document]:
        """Search for documents related to the concept in the text."""
        if not self.docstore:
            raise ValueError("No document store provided.")
        concept = self.identify_concepts(text, self.llm)
        docs = self.docstore.similarity_search(concept, k)
        return docs
    
    def add(self, text: str):
        """Add text to the context."""
        self.context.append(text)
    
    def check(self):
        """If over the limit, identify the concept and return the concept and True."""
        if len(self.context) >= self.limit:
            concepts = self.identify_concept("\n -".join(self.context))
            #TODO: smarter buffer system
            self.context = []
            return True, concepts
        return False, None

class Summarization:
    """Summarize text."""
    def __init__(self, llm=None, limit=5):
        if not llm:
            self.llm = OpenAI()
        else:
            self.llm = llm
        #TODO implement splitting function calls
        self.splitter = CharacterTextSplitter()
        self.count = 0
        self.limit = limit
    
    def summarize(self, text: str, k: int = 4) -> str:
        """Summarize the text."""
        #TODO: add setting
        if self.count < self.limit:
            self.count += 1
            return False, None
        self.count = 0
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Given the following in-progress meeting transcript, summarize the meeting so far in 1-2 sentences:\n {text}")
        prompt = prompt.format(text=text)
        return True, self.llm(prompt)


if __name__ == "__main__":
    c = Summarization(OpenAI())
    test = """
    - The upcoming election will be crucial for the future of the country.
    - There have been many previous elections, but this is particularly important given the candidates.
    - There's a unique opportunity to get out the vote and make a difference locally.
    """
    print(c.summarize(test))
