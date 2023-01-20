from dataclasses import dataclass, field
import datetime

@dataclass
class Segment:
    start: int
    end: int
    text: str

@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    datetime: datetime = datetime.datetime.now()
    meeting_id: str = ''

    def peak(self):
        return self.segments[-1]


@dataclass
class Conversation:
    """
    A dataclass for associating a transcript with the general context and time of the conversation.
        Conversations hold Transcripts which hold segments which bots act on.
    """
    pass

@dataclass
class Trigger:
    #representing text and time based triggers
    input: list[str] | float
    evaluation: str #TODO: switch to enum
    output: bool

@dataclass
class BotConfig:
    trigger: Trigger
    action: str
    name: str

