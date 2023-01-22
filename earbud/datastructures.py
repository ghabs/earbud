from dataclasses import dataclass, field
from langchain import PromptTemplate
import datetime
from enum import Enum
import json
import re

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

#TODO: NB, Enums seem like they overcomplicate things here
class TriggerType(str, Enum):
    """
    An enum for representing the type of trigger.
    """
    CONTAINS = "contains"
    EXACT = "exact"
    REGEX = "regex"
    TIME = "time"
    THRESHOLD = "threshold"

    def __str__(self):
        return self.value

class ActionType(str, Enum):
    """
    An enum for representing the type of action.
    """
    PROMPT = "prompt"
    PREDEFINED = "predefined"
    MATCH = "match"
    SUBSTITUTE = "substitute"

    def __str__(self):
        return self.value

@dataclass
class Trigger:
    #representing text and time based triggers
    input: list[str] | float
    evaluation: TriggerType

    def __call__(self, text: str = None, time: float = None) -> bool:
        """
        Evaluate the trigger.
        """
        if self.evaluation == TriggerType.REGEX:
            #TODO: This is a bit of a hack, but it works for now
            return re.match(self.input[0], text.strip().lower())
        if self.evaluation == TriggerType.CONTAINS:
            return any([word in text for word in self.input])
        elif self.evaluation == TriggerType.EXACT:
            return any([phrase == text for phrase in self.input])
        elif self.evaluation == TriggerType.TIME:
            raise NotImplementedError
        elif self.evaluation == TriggerType.THRESHOLD:
            raise NotImplementedError
        else:
            raise ValueError("Invalid trigger type.")


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
        elif self.type == "match":
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