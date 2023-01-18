# EarBud

Prototype of full system. Will contain a GUI, real time call transcription, summarization, and embeddings search.


## GUI
- TODO: Move from tkinter to PyQT
- TODO: Append transcription to a dialog box.

## Real Time Transcription
Transcribe voice input as it happens.

### Implementation
_1/17_
sd.InputStream calls audio_callback which puts chunks in a queue; a separate thread reads every .5 seconds from the queue and processes and transcribes them.
#### Potential Improvements
- if the last chunk in the audio q is not silence, don't finish the loop; wait an additional chunk until more data comes in and then transcribe.
- determine if time.sleep(0.5) is optimal
- Explore previous_input optimization - maybe it should be a concatenation of n previous text responses, and if silence for more than x attempts, a brand new input.

## TODO: Keyword detection
Take a set of keywords, and if they are in the call/not mentioned after a certain period of time, flag an alert.

## TODO: Summarization
Should summarize parts, or all, of the call.
    - Might need to have speaker diarization


## TODO: Embeddings Search
Take sentences from the call and transcript, turn into embeddings, and then search for relevant docs
    - Will need to filter to highlight 'important' concepts or facts that are amendable to search



## General Product Improvements
- Setting background noise and silence parameters