# Earbud

## Setup
- create a .env file containing an openai key
`touch earbud/.env; echo OPENAI_API_KEY="YOUR_KEY"`
- create a folder for storing transcripts, default file name is .transcripts, editable in gui.py
`mkdir earbud/.transcripts`

- `python3 -m venv .venv`
- `pip3 install -r requirements.txt`

## Running
- `python3 main.py`
Debugging mode is available with --debug

## Misc
docs/ contain the CLA, Legal, and a catalog of todos and useful prompts. By default this is a private closed source project, if your org requires sharing with others/signed nda message the project owner (goldhaber.ben@gmail.com)