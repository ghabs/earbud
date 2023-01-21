# GUI
- [ ] Create figma designs
- [ ] Investigate timeline based designs
- [ ] Determine native app implementation
    - I don't want to invest much time in the GUI until this is decided.
- [ ] Create a progress monitor that tracks queue size and status of the transcription (how far behind it's lagging)
- [ ] Append transcription to a dialog box.
- [ ] BotManager window with customizable params
    - [x] Dispay bots
    - [x] Turn on and off
    - [ ] Display custom params
- [ ] Create a bot notification panel
    - [x] Create the timeline view
    - [ ] Hide/disappear over time
    - [ ] Delineate which bot sends it
- [ ] Investigate peak sound meter (not displaying)
- [ ] Configure based on conversation context

## Design
- [ ] How to handle errors in transcription/bots, what should that... feel like?

# Transcription
- [ ] Top TODO: Explore different blocksizes for faster processing, switch to async
- [ ] Benchmark different model sizes and performance
- [ ] Add a new cache directory for the whisper model
- [ ] On stop, trigger end of transcription
- [ ] set a user location for storing saved transcripts/outputs
- [ ] Context setting: experiment with the benefits of setting a context for the meeting

## Diarization
- [ ] Investigate speaker identification with pyannote; at the very least disambiguate me speaking vs. others speaking.

# Bots

## Keyword Search
- [ ] Reimplement a bot that detects keywords and sends a notification

## Embeddings Search
- [ ] Reimplement a bot that is tied to knowledge graph

## General Purpose Bot
- [ ] Let user create bots from panel

### Implementation
    - A form/parser that takes a user input and turns it into a BotConfig with Trigger and Action.
        - implement enums
    - A general Bot class that takes in the user defined BotConfig.
    - A way of storing and loading the bot config classes

# Robustness
- [ ] Add more debugging features
- [ ] Error handling

# Docs
- [ ] Write up using Blackhole to stream in computer audio.
