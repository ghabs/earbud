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
