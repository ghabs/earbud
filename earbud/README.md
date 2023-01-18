# EarBud

Prototype of full system. Will contain a GUI, real time call transcription, summarization, and embeddings search.

Next Steps:
- Make this very easy to experiment with different bot functions in, and create a testing environment for usability.
    - Top: Bot refactor and bot param setting from GUI
    - setup .wav transcription in the GUI for UX testing
    - Hire a contractor for UI design and native app consultation (1/23)


## GUI
- TODO: Move from tkinter to PyQT
- TODO: Append transcription to a dialog box.
- TODO: BotManager window with customizable params

## Datastructure Refactor

### _datastructures.py_
- Singleton transcript per recording
    - On start of transcription, new transcript is initialized
    - on_callback:
        - Segments are added to the transcript
        - Bots run against the new transcript + the previous segments for context
            - solves the problem of individual bots storing state; decided, bots should be pure functions
    - At end of audio_callback, GUI gets updated with new state of transcript

### Bot Management
- Bots are side-effect free python classes that run against the transcript dataclass
- Parameters are set through the settings interface window
- Bots must have the following
    - a __call__ function that accepts a single piece of text and returns a Tuple of (Bool, str)
        - TODO: decide how to integrate transcript dataclass into _call_; it's probably lightweight enough to be passed to each one, which solves the context problem.
        - TODO: probably implement all default Bot functions as peaking at the top of the segments stack

## Real Time Transcription
Transcribe voice input as it happens.
- TODO: Run a profiler over different transcriptions
        - Test with/without input stream, test memory.

### Implementation
_1/17_
sd.InputStream calls audio_callback which puts chunks in a queue; a separate thread reads every .5 seconds from the queue and processes and transcribes them.
#### Potential Improvements
- if the last chunk in the audio q is not silence, don't finish the loop; wait an additional chunk until more data comes in and then transcribe.
- determine if time.sleep(0.5) is optimal
- Explore previous_input optimization - maybe it should be a concatenation of n previous text responses, and if silence for more than x attempts, a brand new input.

## TODO: Concept Detection
_1/18 In progress_
- Currently summarizes previous k inputs and provides suggestions for concepts embedded
- Next Steps:
    Add the outputs to the concept panel
        - Test disappear over time
    Use the concepts as inputs to VectorStore/DocStore search.


## TODO: Summarization
_1/18 In Progress_
Current summarization function works, but needs experimentation with appropriate timing and context
Should summarize parts, or all, of the call.
    - Might need to have speaker diarization


## TODO: Embeddings Search
Take sentences from the call and transcript, turn into embeddings, and then search for relevant docs
    - Will need to filter to highlight 'important' concepts or facts that are amendable to search

## TODO: Keyword detection
_1/18 In Progress_
Take a set of keywords, and if they are in the call/not mentioned after a certain period of time, flag an alert.
- Need to make this more usable, currently not actually helpful.

## General Product Improvements
- Setting background noise and silence parameters
- Error handling
    - On stop, if the transcription feature hasn't finished, complete transcription or throw a gentle error