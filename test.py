import asyncio
import datetime
import json
import logging
import numpy as np
import os
import queue
import sounddevice as sd
import threading
import timeit
import time
import whisper

from appdirs import user_data_dir
from dotenv import load_dotenv
from langchain import OpenAI
from earbud.bots import BotCreator, SilenceBot
from earbud.datastructures import Transcript, Segment
from earbud.utilities import mtg_summary

load_dotenv()

# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
MODEL_TYPE="base.en"
# pre-set the language to avoid autodetection
LANGUAGE="English"
global_ndarray = None

logging.basicConfig(level=logging.DEBUG,           
					format='%(asctime)s %(message)s',
					datefmt='%m/%d/%Y %I:%M:%S %p')

try:
	logging.debug("Loading model...")
	start = timeit.default_timer()
	model = whisper.load_model(MODEL_TYPE)
	stop = timeit.default_timer()
	logging.debug("Model loaded. %s", stop - start)
except:
	logging.error("Model not found.")


DIR_LOC = os.path.dirname(os.path.realpath(__file__))
TRANSCRIPT_FOLDER = user_data_dir("Earbud")


class Recorder():
	def __init__(self, frontend_fn=lambda x,y: print(x)):
		self.stream = None
		self.recording = False
		self.previously_recording = False
		self.input_overflows = 0
		self.peak = 0
		#TODO check dtype
		self.audio_q = np.ndarray([], dtype=np.float32)
		self.metering_q = queue.Queue()
		self.thread = None
		self.bots = [SilenceBot()]
		self.bot_loading()
		self.transcript = Transcript()
		self.output_fmt = None
		self.mutex = threading.Lock()
		self.n_batch_samples = 10 * 16000
		self.max_queue_size = 3 * self.n_batch_samples
		self.silence_threshold = 0.075
		self.frontend_fn = frontend_fn

	
	def bot_loading(self):
		"""Loads the bots from the bots.json file"""
		bot_config_dir = os.path.join(user_data_dir("earbud", "earbud"), "bot_configs")
		bot_creator = BotCreator()
		try: 
			llm = OpenAI()
		except Exception as e:
			llm = None
			print(e)
		bots = self.bots
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
		chunk: np.ndarray = indata.copy().ravel()
		with self.mutex:
			self.audio_q = np.append(self.audio_q, chunk)

	def process_transcript(self, indata_transformed, prev):
		start = time.time()
		logging.debug("Transcribing, starting %s", start)
		result = model.transcribe(indata_transformed, language=LANGUAGE, condition_on_previous_text=False)
		stop = time.time()
		logging.debug("Function took %s seconds to run", stop - start)
		segments = [Segment(text=s["text"], start=s["start"], end=s["end"],
			temperature=s["temperature"], avg_log_prob=s["avg_logprob"], no_speech_prob=s["no_speech_prob"])
			for s in result['segments']]
		self.transcript.segments.extend(segments)
		result_text = "{text}".format(text=result['text'])
		self.frontend_fn(result_text, "transcript")
		return result['text']

	def process_audio_buffer(self):
		continuation = False
		prev = None
		start = time.time()
		while self.recording:
			self.mutex.acquire()
			if self.audio_q.size >= self.n_batch_samples:
				samples = self.audio_q[:self.n_batch_samples]
				self.audio_q = self.audio_q[self.n_batch_samples:]
				self.mutex.release()
				logging.debug("Processing audio buffer of time %s", time.time() - start)
				start = time.time()
				#If the audio is above the silence threshold, then the speaker is still talking
				#TODO: experiment with multiple thresholds
				if np.amax(samples[-200:-1]) > self.silence_threshold and continuation is False:
					print("continuing")
					continuation = True
					global_ndarray = samples
					continue
				if continuation is True:
					samples = np.concatenate((global_ndarray, samples), dtype=np.float32)
					global_ndarray = None
					continuation = False	
				prev = self.process_transcript(samples, prev)
				"""Bot Management Loop"""
				asyncio.run(self.bot_results())
			else:
				self.mutex.release()

	async def bot_results(self):
		#TODO: add bot results to transcript as they come in and error handling
		for bot in self.bots:
			if bot.active:
				try:
					#TODO: turn bot calls into async
					bot_result = bot(self.transcript)
					"""Update GUI from bot results"""
					if bot_result is not None:
						text = f"{bot.name}: {bot_result}"
						self.frontend_fn({"name": bot.name, "text": text}, "bot")
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
			self.frontend_fn(result, "output")
		except Exception as e:
			print(e)
	
	def silence_callback(self, indata, frames, time, status):
		silence_max = np.max(np.abs(indata.ravel()))
		self.silence_threshold = max(silence_max, self.silence_threshold) * 5

	
	def set_silence_threshold(self):
		self.stream = sd.InputStream(samplerate=16000, dtype='float32', channels=1, callback=self.silence_callback)
		self.stream.start()

	def record(self):
		print("Recording")
		if self.stream is not None:
			self.stream.close()
		self.stream = sd.InputStream(samplerate=16000, dtype='float32', channels=1, callback=self.audio_callback)
		self.stream.start()
		self.recording = True
		self.thread = threading.Thread(target=self.process_audio_buffer)
		self.thread.start()
	
	def stop(self):
		self.recording = False
		self.stream.close()
		self.stream = None
		self.thread.join()

if __name__ == '__main__':
	recorder = Recorder()
	recorder.record()

