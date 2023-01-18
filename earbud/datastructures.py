from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class Transcript:
    segments: List[Segment]
    datetime: str
    meeting_id: Optional(str)
