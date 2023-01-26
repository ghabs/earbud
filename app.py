import eel
from test import Recorder
from earbud.bots import BotCreator
from earbud.output_fmts import user_output_fmts, save_output_fmt

recorder = Recorder()

@eel.expose
def record_py():
	recorder.record()

@eel.expose
def stop_py():
	print("Stopping")
	recorder.recording = False
	recorder.stream.close()
	recorder.stream = None
	print("Stopped Recording")

@eel.expose
def format_output_py():
	recorder.format_output()

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
		return eel.appendBotText(msg)
	if caller == "output":
		return eel.setOutputText(msg)
	return print(msg)

recorder.frontend_fn = frontend_fn

if __name__ == '__main__':
	eel.init('fe')
	eel.start('main.html')