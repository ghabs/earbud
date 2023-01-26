import asyncio
import datetime
import eel
import json
import logging
import numpy as np
import os
import queue
import sounddevice as sd
import threading
import timeit
import whisper

from appdirs import user_data_dir
from dotenv import load_dotenv
from langchain import OpenAI
from earbud.bots import ConceptBot, SummarizeBot, BotCreator
from earbud.datastructures import Transcript, Segment
from earbud.utilities import mtg_summary
from earbud.output_fmts import user_output_fmts, save_output_fmt


load_dotenv()

# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
MODEL_TYPE="tiny.en"
# pre-set the language to avoid autodetection
LANGUAGE="English"
global_ndarray = None
logger = logging.getLogger()

try:
	logger.debug("Loading model...")
	start = timeit.default_timer()
	model = whisper.load_model(MODEL_TYPE)
	stop = timeit.default_timer()
	logger.debug("Model loaded. %s", stop - start)
except:
	logger.error("Model not found.")


SILENCE_THRESHOLD=400
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=100

DIR_LOC = os.path.dirname(os.path.realpath(__file__))
TRANSCRIPT_FOLDER = user_data_dir("Earbud")


class Recorder():
	def __init__(self):
		self.stream = None
		self.recording = False
		self.previously_recording = False
		self.input_overflows = 0
		self.peak = 0
		self.audio_q = queue.Queue()
		self.metering_q = queue.Queue()
		self.thread = None
		self.bots = []
		self.bot_loading()
		self.transcript = Transcript()
		self.output_fmt = None

	
	def bot_loading(self):
		"""Loads the bots from the bots.json file"""
		bot_config_dir = os.path.join(user_data_dir("earbud", "earbud"), "bot_configs")
		bot_creator = BotCreator()
		try: 
			llm = OpenAI()
		except Exception as e:
			llm = None
			print(e)
		bots = [ConceptBot(llm=llm), SummarizeBot(llm=llm),]
		if not os.path.exists(bot_config_dir):
			os.makedirs(bot_config_dir)
		try:
			for bot_file in os.listdir(bot_config_dir):
				if bot_file.endswith(".json"):
					bot_file_json = json.load(open(os.path.join(bot_config_dir, bot_file)))
					bots.append(bot_creator.load_bot_config(bot_file_json))
			self.bots = bots
		except:
			print("error loading bots")
			self.bots = []

	def audio_callback(self, indata, frames, time, status):
		"""This is called (from a separate thread) for each audio block."""
		if status.input_overflow:
			# NB: This increment operation is not atomic, but this doesn't
			#     matter since no other thread is writing to the attribute.
			self.input_overflows += 1
		# NB: self.recording is accessed from different threads.
		#     This is safe because here we are only accessing it once (with a
		#     single bytecode instruction).
		if self.recording:
			self.audio_q.put(indata.copy())
			self.previously_recording = True
		else:
			if self.previously_recording:
				self.audio_q.put(None)
				self.previously_recording = False

		self.peak = max(self.peak, np.max(np.abs(indata)))
		try:
			self.metering_q.put_nowait(self.peak)
		except queue.Full:
			pass
		else:
			self.peak = 0

	def process_transcript(self, indata_transformed, prev):
		if logger.level == logging.DEBUG:
			start = timeit.default_timer()
			logger.debug("Transcribing, starting %s", start)
		result = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=prev)
		if logger.level == logging.DEBUG:
			stop = timeit.default_timer()
			logger.debug("Time: %s", stop - start)
		segments = [Segment(text=s["text"], start=s["start"], end=s["end"],
			temperature=s["temperature"], avg_log_prob=s["avg_logprob"], no_speech_prob=s["no_speech_prob"])
			for s in result['segments']]
		self.transcript.segments.extend(segments)
		result_text = "{text}".format(text=result['text'])
		eel.appendTranscriptText(result_text)
		return result['text']

	def process_audio_buffer(self):
			global global_ndarray
			global_ndarray = None
			queue = self.audio_q
			continuation = False
			prev = None
			while self.recording:
				eel.sleep(1)
				#TODO: determine if sleep function when no talking, make sure not overloading
				while queue.empty() is False:
					indata = queue.get()
					#TODO: handle none case, determine if global_ndarray is cleared
					try:
						indata_flattened = abs(indata.flatten())
					except AttributeError as e:
						print(e)
						continue

					# discard buffers that contain mostly silence only if previous buffer wasn't a continuation
					if(continuation is False and (np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO)):
						continue
							
					if (global_ndarray is not None):
						global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
					else:
						global_ndarray = indata
					
					#TODO: might not be robust to general streaming audio, better to use a buffer and go back and fix earlier mistakes
					continuation = True
					# keep recording if the person is still talking
					if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/5):
						continue
					else:
						local_ndarray = global_ndarray.copy()
						global_ndarray = None
						indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
						"""Transcribe Audio"""
						print(indata_transformed)
						prev = self.process_transcript(indata_transformed, prev)
						"""Bot Management Loop"""
						#TODO: make sure only called once
						asyncio.run(self.bot_results())
						continuation = False
						del local_ndarray
						del indata_flattened

	async def bot_results(self):
		#TODO: add bot results to transcript as they come in and error handling
		for bot in self.bots:
			if bot.active:
				try:
					#TODO: turn bot calls into async
					bot_result = bot(self.transcript)
					"""Update GUI from bot results"""
					if bot_result is not None:
						print("firing bot")
						text = f"{bot.name}: {bot_result}"
						eel.appendBotText({"name": bot.name, "text": text})
				except Exception as e:
					print(e)
					print("Error with bot: " + str(bot))

	def write_transcript(self, transcript: Transcript):
		date = None
		try:
			date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			with open(f'{TRANSCRIPT_FOLDER}/{date}_{MODEL_TYPE}_transcript.txt', 'w') as f:
				for s in transcript.segments:
					f.write(s.text + '\n')
		except Exception as e:
			print(f'Error occurred while writing transcript {date}: {e}')
	
	def format_output(self):
		#TODO set as async
		prompt = """Given the following transcript of a single person talking, create a summary doc that contains the following -
					Summary: 1-2 sentences describing what the person was talking about
					Action Items: A list of action items the person said

					Transcript: {transcript}"""
		try:
			llm = OpenAI(temperature=0.0)
			result = mtg_summary(self.transcript, prompt, llm)
			print(result)
			eel.setOutputText(result)
		except Exception as e:
			print(e)
		

recorder = Recorder()

"""
Eel Exposed Functions for Frontend
"""

@eel.expose
def record_py():
	print("Recording")
	#TODO: set input device
	def create_stream(device=None):
		if recorder.stream is not None:
			recorder.stream.close()
		recorder.stream = sd.InputStream(samplerate=16000, dtype='int16',
			device=device, channels=1, callback=recorder.audio_callback)
		recorder.stream.start()
	create_stream()
	recorder.recording = True
	recorder.thread = threading.Thread(target=recorder.process_audio_buffer)
	recorder.thread.start()

@eel.expose
def stop_py():
	recorder.recording = False
	recorder.stream.close()
	recorder.stream = None
	print("Stopped Recording")

@eel.expose
def format_output_py():
	print("Formatting Transcript")
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

if __name__ == '__main__':
	eel.init('fe')
	eel.start('main.html')
