from typing import List
from llm import ConceptSearcher
from langchain import OpenAI, PromptTemplate, LLMChain

from datastructures import Transcript, Segment

class Bot():
    """
    The base class for all bots.
    #TODO: API representation for adding text to bots/determine global text datastructure
    """
    def __init__(self) -> None:
        self.active = True
    
    def _empty_segment(self, transcript: Transcript) -> bool:
        """
        Check if the most recent transcript segment is empty.
        """
        return transcript.peak().text == ""

    def __call__(self, text: str) -> str:
        """
        Run the bot on the text.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """
        Return a string representation of the bot.
        """
        return f"{self.__class__.__name__}"


class BotManager():
    """
    Hold bots and manage them for the main GUI
    GUI: Display the different bots and allow a user to set the params in settings panel
    """
    pass


class KeywordBot(Bot):
    """
    A bot that searches for keywords in a sentence and returns the keyword.
    """
    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords
        super().__init__()

    def __call__(self, text: str) -> str:
        """
        Run the bot on the text.
        """
        for keyword in self.keywords:
            if keyword in text:
                return keyword
        return None

class ConceptBot(Bot):
    def __init__(self, llm, k=1) -> None:
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="From the following bullet points, identify 1 to 3 key concepts where more research could be useful:\n {text}")
        super().__init__()
        self.active = False
    
    def _concepts(self, segments:List[Segment]):
        segs = segments[-self.k:]
        text = " ".join([seg.text for seg in segs])
        prompt = self.prompt.format(text=text)
        return self.llm(prompt)
    
    def __call__(self, transcript:Transcript) -> str:
        """
        Run the bot on the text.
        """
        if not self._empty(transcript):
            return None
        return self._concepts(transcript.segments)

class SummarizeBot(Bot):
    def __init__(self, llm, k=1, window = 5) -> None:
        self.llm = llm
        self.k = k
        self.window = window
        self.i = 0
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="Given the following in-progress meeting transcript, summarize the meeting so far in 1-2 sentences:\n {text}")
        super().__init__()
    
    def _summarize(self, segments:List[Segment]):
        segs = segments[-self.k:]
        text = " ".join([seg.text for seg in segs])
        prompt = self.prompt.format(text=text)
        return self.llm(prompt)
    
    def __call__(self, transcript:Transcript) -> str:
        """
        Run the bot on the text.
        """
        # Only summarize if the transcript is empty and the unsummarized transcript length is a multiple of self.tl
        segments = transcript.segments[self.i:]
        if self._empty_segment(transcript) or len([True for s in segments if s.text != ""]) < self.window:
            return None
        self.i = len(transcript.segments) - 1
        return self._summarize(transcript.segments)
