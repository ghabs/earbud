from datastructures import Transcript, Segment
from langchain import PromptTemplate
def text_clean(text:str) -> str:
    """
    Clean the text segment.
    """
    return text.strip()


def improve_transcript_text(segments: list[Segment]) -> PromptTemplate:
    """
    Create a prompt template for improving the template
    #TODO: Measure time for external calls
    """
    text = " ".join([text_clean(seg.text) for seg in segments])
    prompt = """This meeting transcript may, or may not, have some minor errors in it. Go through and, if a word or phrase or sentence seems out of place, correct it with your best guess for what was actually being said.
    Transcript: {text}
    Corrected Transcript:
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt)
    prompt = prompt.format(text=text)
    return prompt