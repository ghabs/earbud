import eel
from main import Realtime_Whisper
from earbud.bots import BotCreator
from earbud.output_fmts import user_output_fmts, save_output_fmt
from langchain import OpenAI
from earbud.utilities import mtg_summary
from earbud.output_fmts import user_output_fmts, save_output_fmt
from dotenv import load_dotenv
load_dotenv()

recorder = Realtime_Whisper("base.en")

class OutputFormat:
	def __init__(self, name, value):
		self.name = name
		self.output_fmt = value
		self.transcript = None
	
	def formatted_transcript(self, **kwds):
		try:
			llm = OpenAI(temperature=0.0)
			result = mtg_summary(self.transcript, self.output_fmt, llm)
			return result
		except Exception as e:
			print(e)

formatter = OutputFormat("mtg", user_output_fmts()[0]["value"])

@eel.expose
def record_py():
	print("Recording")
	recorder.start()

@eel.expose
def stop_py():
	recorder.stop()

@eel.expose
def format_output_py():
	print("Formatting Transcript")
	return formatter.formatted_transcript()


@eel.expose
def save_py():
	try:
		recorder.write_transcript(recorder.transcript)
		eel.clearTranscript()
	except Exception as e:
		print(e)

@eel.expose
def get_bots_py():
	return [{"name": bot.name, "active": bot.active} for bot in recorder.bots]

@eel.expose
def toggle_bot_py(name):
	for bot in recorder.bots:
		if bot.name == name:
			bot.active = not bot.active
			break
	return get_bots_py()

@eel.expose
def create_bot_py(name, trigger, trigger_value, action, action_value):
	print("Creating Bot")
	bot_creator = BotCreator()
	bot_config = bot_creator.make_bot_config(name, trigger, trigger_value, action, action_value)
	recorder.bots.append(bot_creator.create(bot_config))
	bot_creator.store_bot_config(bot_config)
	return get_bots_py()

@eel.expose
def output_fmts_py():
	return user_output_fmts()

@eel.expose
def set_output_format_py(fmt):
	print(f"Setting Output Format {fmt}")
	recorder.output_fmt = fmt

@eel.expose
def create_output_format_py(fmt):
	print(f"Creating Output Format {fmt}")
	save_output_fmt(fmt)
	return output_fmts_py()


def frontend_fn(msg, caller=None):
	"""
		Used to pass text from backend to frontend
		caller - name of calling fn (string) used to route to eel fn
		msg - message (dict)
	"""
	print(f"Frontend Fn: {caller} - {msg}")
	if caller == "transcript":
		return eel.appendTranscriptText(msg)
	if caller == "bot":
		return eel.bot_feed(msg)
	if caller == "set_output":
		formatter.transcript = msg
		eel.setOutputText(msg)
	if caller == "output":
		return eel.setOutputText(msg)
	return print(msg)

recorder.stream_fn = frontend_fn

if __name__ == '__main__':
	eel.init('fe')
	eel.start('main.html')