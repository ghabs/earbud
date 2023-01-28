# Top
- [ ] Fix hallucination
    - https://github.com/openai/whisper/discussions/679
        - https://github.com/openai/whisper/discussions/679#discussioncomment-4540909
    - https://github.com/snakers4/silero-vad/blob/master/LICENSE
        - VAD to detect 

# GUI
- [x] Create figma designs
- [x] Investigate timeline based designs
- [x] Determine native app implementation
    - I don't want to invest much time in the GUI until this is decided.
- [ ] Create a progress monitor that tracks queue size and status of the transcription (how far behind it's lagging)
- [x] Investigate greenlet vs thread with eel.
    - pretty sure I want to minimize threads created overall and new approach does that.
- [ ] Append transcription metadata (filename, context, conversation)
- [ ] BotManager window with customizable params
    - [x] Dispay bots
    - [x] Turn on and off
    - [ ] Display custom params
- [ ] Create a bot notification panel
    - [x] Create the timeline view
    - [ ] Hide/disappear over time
    - [x] Add bot name
- [ ] Configure based on conversation context

- [ ] Separate frontend code from recorder code
    - format_output pass text back to front end
    - process_transcript, bot_results should accept eel.appendText as a callback param
        - maybe a generic one and the specific place the text goes is strictly frontend
    - Maybe Create one general use function for calling JS eel that pases them back 
    


## Design
- [ ] How to handle errors in transcription/bots, what should that... feel like?

# Transcription
- [x] Explore different blocksizes for faster processing
    - investigate more roubst, transcription tied to the bot design/switch to async
    - [x] Test whether new method works better/more reliably
        - Much cleaner code implementation, unclear if it works that much better. still getting some 'imagining'
        - Longer form audio segments that get closer to 30 seconds work better.
- [ ] Explore noise suppression 
    - https://github.com/xiph/rnnoise
- [ ] Emulate full file transcription, detection of significant pauses.
    - Detect silence and use that to trigger transcription, unless it's over the limit of the queue
- [ ] Add ambient noise filtering
- [ ] Benchmark different model sizes and performance
- [ ] Add a new cache directory for the whisper model
- [ ] Faster load time of initial transcription
- [ ] On stop, trigger end of transcription
- [x] set a user location for storing saved transcripts/outputs
  - [ ] let a user specify through filedrop down where it's saved
- [ ] Context setting: experiment with the benefits of setting a context for the meeting
- [ ] investigate calling an llm on transcript to improve quality

## Output Format
- [ ] Try out different prompts
- [ ] Reduce the risk of imagineering. It should only put in what was in the text.

## Diarization
- [x] Investigate speaker identification with pyannote; at the very least disambiguate me speaking vs. others speaking.
    - Using embeddings from speechbrain to verify
- [ ] Implement the Diarization feature
- [ ] Test whisper diarization

# Bots
- [ ] add a user profile so if a user deactivates or removes a bot from their user dir its not readded.
- [ ] Push certain bot outputs to the transcript

## Keyword Search
- [ ] Reimplement a bot that detects keywords and sends a notification

## Embeddings Search
- [ ] Reimplement a bot that is tied to knowledge graph

## General Purpose Bot
- [x] Let user create bots from panel

### Implementation
- [x] A form/parser that takes a user input and turns it into a BotConfig with Trigger and Action.
    - implement enums
- [x] A general Bot class that takes in the user defined BotConfig.
- [x] A way of storing and loading the bot config classes

# Robustness
- [ ] Faster load time through lazy loading in GUI
- [ ] Add more debugging features
- [ ] Error handling

# Docs
- [ ] Write up using Blackhole to stream in computer audio.
- [ ] Write vision doc
- [ ] Add Design sketches
