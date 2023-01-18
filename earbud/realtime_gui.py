#!/usr/bin/env python3
"""Simple GUI for recording into a WAV file.

There are 3 concurrent activities: GUI, audio callback, file-writing thread.

Neither the GUI nor the audio callback is supposed to block.
Blocking in any of the GUI functions could make the GUI "freeze", blocking in
the audio callback could lead to drop-outs in the recording.
Blocking the file-writing thread for some time is no problem, as long as the
recording can be stopped successfully when it is supposed to.

"""
import contextlib
import queue
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.simpledialog import Dialog

import numpy as np
import sounddevice as sd
import soundfile as sf

import whisper

import queue
import sys
import time 

from keywords import KeywordSearcher


# SETTINGS
MODEL_TYPE="base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
BLOCKSIZE=24678 
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds. 

global_ndarray = None
model = whisper.load_model(MODEL_TYPE)
SILENCE_THRESHOLD=400
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=100


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
		self.rec_button.grid(row=0, column=0, padx=padding, pady=padding)
		self.rec_button.pack(side='left', padx=padding, pady=padding, anchor='w')

		self.settings_button = ttk.Button(
			f, text='settings', command=self.on_settings)
		self.settings_button.pack(side='left', padx=padding, pady=padding, anchor='w')

		f.pack(expand=True, padx=padding, pady=padding)

		self.keywords = ttk.Entry()
		self.keywords.pack()

		self.input_overflows = 0
		self.status_label = ttk.Label()
		self.status_label.pack(anchor='w')

		self.meter = ttk.Progressbar()
		self.meter['orient'] = 'horizontal'
		self.meter['mode'] = 'determinate'
		self.meter['maximum'] = 1.0
		self.meter.pack(fill='x')
		
		self.transcibed_window = tk.Text(f, height=10, width=50, wrap=tk.WORD)
		self.transcibed_window.pack()

		self.search_window = tk.Text(f, height=10, width=50, wrap=tk.WORD)
		self.search_window.pack()

		self.keyword_searcher = KeywordSearcher(self.keywords.get())

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

	def process_audio_buffer(self):
		global global_ndarray
		global_ndarray = None
		prev = ''
		queue = self.audio_q

		while self.recording:
			time.sleep(0.5)
			while queue.empty() is False:
				indata = queue.get()
				indata_flattened = abs(indata.flatten())
				
				# discard buffers that contain mostly silence
				if((np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO) and (global_ndarray is None)):
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
					result = model.transcribe(indata_transformed, language=LANGUAGE, initial_prompt=prev)
					self.transcibed_window.insert(tk.END, "\n -")
					self.transcibed_window.insert(tk.END, result['text'])
					_, keycheck = self.check_keywords(result['text'], self.keywords.get().split(','))
					if _:
						self.transcibed_window.insert(tk.END, "\n")
						self.transcibed_window.insert(tk.END, "(Found keyword: " + keycheck + ")")
					#TODO: don't constantly search, only search when an important word is found
					#search_results = self.keyword_searcher.search_docs(result['text'])
					#if search_results:
				#		for result in search_results:
				#			self.search_window.insert(tk.END, "\n -")
				#			self.search_window.insert(tk.END, result)
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
	app = RecGui()
	app.mainloop()


if __name__ == '__main__':
	main()