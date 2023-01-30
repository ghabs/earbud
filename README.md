# Earbud

## Setup
- create a .env file containing an openai key
`touch earbud/.env; echo OPENAI_API_KEY="YOUR_KEY"`
- create a folder for storing transcripts, default file name is .transcripts, editable in gui.py
`mkdir earbud/.transcripts`

- `python3 -m venv .venv`
- `pip3 install -r requirements.txt`

## Running
- `python3 app.py`
Note: There's a delay built in around the recording feature of five seconds for models to load, idk if that's needed anymore.

## Misc
docs/ contain the CLA, Legal, and a catalog of todos and useful prompts. By default this is a private closed source project, if your org requires sharing with others/signed nda message the project owner (goldhaber.ben@gmail.com)
