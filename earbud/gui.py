#!/usr/bin/env python3
"""GUI for transcription and bot interaction.
"""
import asyncio
import argparse
import contextlib
import datetime
import logging
import numpy as np
import os
import queue
import sounddevice as sd
import soundfile as sf
import time
import timeit 
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import Dialog
import whisper

from appdirs import user_data_dir
from bot_panel import BotPanel
from bots import ConceptBot, SummarizeBot
from datastructures import Transcript, Segment
from langchain import OpenAI

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(format='%(filename)s %(asctime)s %(message)s')
logger = logging.getLogger()



# SETTINGS
MODEL_TYPE="base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
BLOCKSIZE=24678 
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds. 

global_ndarray = None
#TODO: add lazy loading
try:
	#TODO: set debug level from global config and main.py
	logging.debug("Loading model...")
	start = timeit.default_timer()
	model = whisper.load_model(MODEL_TYPE)
	stop = timeit.default_timer()
	logging.debug("Model loaded. %s", stop - start)
except:
	logger.error("Model not found.")

SILENCE_THRESHOLD=400
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=100

DIR_LOC = os.path.dirname(os.path.realpath(__file__))
TRANSCRIPT_FOLDER = user_data_dir("Earbud")


class SettingsWindow(Dialog):
	"""Dialog window for choosing sound device."""

	def body(self, master):
		ttk.Label(master, text='Select host API:').pack(anchor='w')
		self.hostapi_list = ttk.Combobox(master, state='readonly', width=50)
		self.hostapi_list.pack()
		self.hostapi_list['values'] = [
			hostapi['name'] for hostapi in sd.query_hostapis()]

		ttk.Label(master, text='Select sound device:').pack(anchor='w')
		self.device_ids = []
		self.device_list = ttk.Combobox(master, state='readonly', width=50)
		self.device_list.pack()

		self.hostapi_list.bind('<<ComboboxSelected>>', self.update_device_list)
		with contextlib.suppress(sd.PortAudioError):
			self.hostapi_list.current(sd.default.hostapi)
			self.hostapi_list.event_generate('<<ComboboxSelected>>')

	def update_device_list(self, *args):
		hostapi = sd.query_hostapis(self.hostapi_list.current())
		self.device_ids = [
			idx
			for idx in hostapi['devices']
			if sd.query_devices(idx)['max_input_channels'] > 0]
		self.device_list['values'] = [
			sd.query_devices(idx)['name'] for idx in self.device_ids]
		default = hostapi['default_input_device']
		if default >= 0:
			self.device_list.current(self.device_ids.index(default))

	def validate(self):
		self.result = self.device_ids[self.device_list.current()]
		return True


class RecGui(tk.Tk):

	stream = None

	def __init__(self):
		super().__init__()

		self.title('Recording GUI')

		padding = 10

		f = ttk.Frame()

		self.rec_button = ttk.Button(f)
		self.rec_button.pack(side='left', padx=padding, pady=padding, anchor='n')

		self.settings_button = ttk.Button(
			f, text='settings', command=self.on_settings)
		self.settings_button.pack(side='left', padx=padding, pady=padding, anchor='n')

		self.finish_button = ttk.Button(
			f, text='finish', command=self.on_finish)
		self.finish_button.pack(side='left', padx=padding, pady=padding, anchor='n')

		self.bot_panel_button = ttk.Button(
			f, text='bot panel', command=self.on_bot_panel)
		self.bot_panel_button.pack(side='left', padx=padding, pady=padding, anchor='n')

		f.pack(expand=True, padx=padding, pady=padding)

		self.transcript = Transcript()

		self.input_overflows = 0
		self.status_label = ttk.Label(f)
		self.status_label.pack(anchor='w')

		self.meter = ttk.Progressbar()
		self.meter['orient'] = 'horizontal'
		self.meter['mode'] = 'determinate'
		self.meter['maximum'] = 1.0
		self.meter.pack(fill='x')

		wf = ttk.Frame()
		wf.pack(fill='both', expand=True, padx=padding, pady=padding)
		self.transcribed_window = tk.Text(wf, height=100, width=30, wrap=tk.WORD)
		self.transcribed_window.pack(side=tk.LEFT)

		self.bot_window = tk.Text(wf, height=100, width=30, wrap=tk.WORD)
		self.bot_window.pack(side=tk.RIGHT)

		#TODO: these should be lazy loaded to reduce time to load, refactor
		try:
			llm = OpenAI()
		except:
			print("Error loading LLM")
		self.bots = [ConceptBot(llm=llm), SummarizeBot(llm=llm),]
		# We try to open a stream with default settings first, if that doesn't
		# work, the user can manually change the device(s)
		self.create_stream()

		self.recording = self.previously_recording = False
		self.audio_q = queue.Queue()
		self.peak = 0
		self.metering_q = queue.Queue(maxsize=1)

		self.protocol('WM_DELETE_WINDOW', self.close_window)
		self.init_buttons()
		self.update_gui()
	
	def check_keywords(self, text, keywords):
		text = text.lower()
		text = text.replace(".", "")
		text = text.replace(",", "")
		text = text.replace("!", "")
		text = text.replace("?", "")
		text = text.split(" ")
		for keyword in keywords:
			if keyword in text:
				return True, keyword
		return False, None
	
	async def bot_results(self):
		bot_results = [(str(bot), bot(self.transcript)) for bot in self.bots if bot.active]
		"""Update GUI from bot results"""
		for bot_result in bot_results:
			if bot_result[1] is not None:
				self.bot_window.insert(tk.END, f"\n")
				self.bot_window.insert(tk.END, bot_result)

	def process_audio_buffer(self):
		global global_ndarray
		global_ndarray = None
		prev = ''
		queue = self.audio_q
		#TODO: sometimes with long periods of silence, a random transcription is fired.
		#  Not sure why, probably has to do with chunking and the global_ndarray

		while self.recording:
			#TODO: determine if sleep function when no talking, make sure not overloading
			while queue.empty() is False:
				indata = queue.get()
				try:
					indata_flattened = abs(indata.flatten())
				except AttributeError as e:
					print(e)
					continue
				
				# discard buffers that contain mostly silence
				if((np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO)):
					continue
						
				if (global_ndarray is not None):
					global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
				else:
					global_ndarray = indata
				#TODO: need a better background sound parameter
				if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/5):
					continue
				else:
					local_ndarray = global_ndarray.copy()
					global_ndarray = None
					indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
					if logger.level == logging.DEBUG:
						start = timeit.default_timer()
						logger.debug("Transcribing, starting %s", start)
					result = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=prev)
					if logger.level == logging.DEBUG:
						stop = timeit.default_timer()
						logger.debug("Time: %s", stop - start)
					#TODO get start and end times from result.segment
					seg = Segment(text=result['text'].strip(), start=0, end=0)
					self.transcript.segments.append(seg)
					#TODO: Add a separate update GUI function that reflects the transcript, handles trim(), etc.
					if seg.text != '':
						self.transcribed_window.insert(tk.END, "\n\n")
						self.transcribed_window.insert(tk.END, result['text'])
					"""Bot Management Loop"""
					asyncio.run(self.bot_results())

				prev = result["text"]
					
				del local_ndarray
				del indata_flattened

	def create_stream(self, device=None):
		if self.stream is not None:
			self.stream.close()
		self.stream = sd.InputStream(samplerate=16000, dtype='int16', blocksize=BLOCKSIZE,
			device=device, channels=1, callback=self.audio_callback)
		self.stream.start()

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
	
	def write_transcript(self, transcript: Transcript):
		date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		with open(f'{TRANSCRIPT_FOLDER}/{date}_{MODEL_TYPE}_transcript.txt', 'w') as f:
			for s in transcript.segments:
				f.write(s.text + '\n')

	def on_rec(self):
		self.settings_button['state'] = 'disabled'
		self.recording = True

		if self.audio_q.qsize() != 0:
			print('WARNING: Queue not empty!')
		self.thread = threading.Thread(
			target=self.process_audio_buffer
		)
		self.thread.start()

		self.rec_button['text'] = 'stop'
		self.rec_button['command'] = self.on_stop
		self.rec_button['state'] = 'normal'

	def on_stop(self, *args):
		self.rec_button['state'] = 'disabled'
		self.recording = False
		self.wait_for_thread()
	
	def on_finish(self, *args):
		self.wait_for_thread()
		self.write_transcript(self.transcript)
		self.transcribed_window.delete('1.0', tk.END)
		self.transcript = Transcript()

	def wait_for_thread(self):
		# NB: Waiting time could be calculated based on stream.latency
		self.after(10, self._wait_for_thread)

	def _wait_for_thread(self):
		if self.thread.is_alive():
			self.wait_for_thread()
			return
		self.thread.join()
		self.init_buttons()

	def on_settings(self, *args):
		w = SettingsWindow(self, 'Settings')
		if w.result is not None:
			self.create_stream(device=w.result)
	
	def on_bot_panel(self, *args):
		w = BotPanel(self, 'Bot Panel', self.bots)
		if w.result is not None:
			self.bots = w.result

	def init_buttons(self):
		self.rec_button['text'] = 'record'
		self.rec_button['command'] = self.on_rec
		if self.stream:
			self.rec_button['state'] = 'normal'
		self.settings_button['state'] = 'normal'

	def update_gui(self):
		try:
			peak = self.metering_q.get_nowait()
		except queue.Empty:
			pass
		else:
			self.meter['value'] = peak
		self.after(100, self.update_gui)

	def close_window(self):
		if self.recording:
			self.on_stop()
		self.destroy()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', action='store_true',
						help='Enables debug mode', default=False)
	args = parser.parse_args()
	if args.debug:
		logger.setLevel(logging.DEBUG)
	app = RecGui()
	app.mainloop()


if __name__ == '__main__':
	main()