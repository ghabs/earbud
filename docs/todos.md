# GUI
- [ ] Create figma designs
- [ ] Investigate timeline based designs
- [ ] Determine native app implementation
    - I don't want to invest much time in the GUI until this is decided.
- [ ] Create a progress monitor that tracks queue size and status of the transcription (how far behind it's lagging)
- [ ] Append transcription metadata (filename, context, conversation)
- [ ] BotManager window with customizable params
    - [x] Dispay bots
    - [x] Turn on and off
    - [ ] Display custom params
- [ ] Create a bot notification panel
    - [x] Create the timeline view
    - [ ] Hide/disappear over time
    - [x] Add bot name
- [ ] Investigate peak sound meter (not displaying)
- [ ] Configure based on conversation context

## Design
- [ ] How to handle errors in transcription/bots, what should that... feel like?

# Transcription
- [x] Explore different blocksizes for faster processing
    - investigate more roubst, transcription tied to the bot design/switch to async
        - Emulate full file transcription, detection of significant pauses.
- [ ] Benchmark different model sizes and performance
- [ ] Add a new cache directory for the whisper model
- [ ] On stop, trigger end of transcription
- [x] set a user location for storing saved transcripts/outputs
  - [ ] let a user specify through filedrop down where it's saved
- [ ] Context setting: experiment with the benefits of setting a context for the meeting
- [ ] investigate calling an llm on transcript to improve quality

## Diarization
- [ ] Investigate speaker identification with pyannote; at the very least disambiguate me speaking vs. others speaking.

# Bots
- [ ] add a user profile so if a user deactivates or removes a bot from their user dir its not readded.

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
