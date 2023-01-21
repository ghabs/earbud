from dataclasses import dataclass, field
from langchain import PromptTemplate
import datetime
from enum import Enum
import json

@dataclass
class Segment:
    start: float
    end: float
    text: str
    temperature: float| None = None
    avg_log_prob: float| None = None
    no_speech_prob: float| None = None

@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    datetime: datetime = datetime.datetime.now()
    meeting_id: str = ''

    def peak(self):
        if len(self.segments) == 0:
            return Segment(start=0, end=0, text='')
        return self.segments[-1]


@dataclass
class Conversation:
    """
    A dataclass for associating a transcript with the general context and time of the conversation.
        Conversations hold Transcripts which hold segments which bots act on.
    """
    pass

class TriggerType(Enum):
    """
    An enum for representing the type of trigger.
    """
    TEXT = 1
    TIME = 2

@dataclass
class Trigger:
    #representing text and time based triggers
    input: list[str] | float
    evaluation: str #TODO: switch to enum

    def __call__(self, text: str = None, time: float = None) -> bool:
        """
        Evaluate the trigger.
        """
        if isinstance(self.input, list):
            if self.evaluation == "contains":
                return any([word in text for word in self.input])
            elif self.evaluation == "exact":
                return any([phrase == text for phrase in self.input])
            else:
                raise NotImplementedError
        elif isinstance(self.input, float):
            if self.evaluation == "time":
                return time < self.input
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

class ActionType(Enum):
    """
    An enum for representing the type of action.
    """
    PROMPT = 1
    PREDEFINED = 2

class Action:
    """
    A class for representing actions that bots can take.
    """
    def __init__(self, type:str, action: str) -> None:
        self.type = type
        self.action = action

    def __call__(self) -> str:
        """
        Run the action on the text.
        """
        if self.type == "prompt":
            return PromptTemplate(input_variables=["text"], template=self.action)
        elif self.type == "predefined":
            return self.action
        else:
            raise NotImplementedError
    
    def __repr__(self) -> str:
        """
        Return a string representation of the action.
        """
        return f"{self.type}: {self.action}"

@dataclass
class BotConfig:
    trigger: Trigger
    action: Action
    name: str

    def _to_json(self) -> str:
        """
        Return a json representation of the bot.
        """
        d = self.__dict__
        d["trigger"] = d["trigger"].__dict__
        d["action"] = d["action"].__dict__
        return json.dumps(d)