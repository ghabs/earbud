# asr using whisper and Silero-VAD (https://github.com/snakers4/silero-vad)
# structure based on the very nice work of Oliver Guhr over at https://github.com/oliverguhr/wav2vec2-live
import asyncio
import pyaudio
import numpy as np
import threading
import time
from sys import exit
from queue import Queue
import matplotlib.pylab as plt
import wave
import whisper
import struct
import multiprocessing
import torch
import re

from earbud.bots import KeywordBot

# for Debugging: save the audiostream that was provided to whisper after sending through queue
filename = 'audio_provided.wav'
# for Debugging: save the audiostream that was actually recorded pre sending.
filename_orig = 'audio_recorded.wav'

def check_string(string):
    pattern = r'^\.+$'
    if re.search(pattern, string):
        return True
    else:
        return False

async def bot_feed(bots, segments, stream_fn):
    # TODO: add bot results to transcript as they come in and error handling
    for bot in bots:
        if bot.active:
            try:
                # TODO: turn bot calls into async
                bot_result = bot(segments)
                """Update GUI from bot results"""
                if bot_result is not None:
                    text = f"{bot.name}: {bot_result}"
                    stream_fn({"name": bot.name, "text": text}, "bot")
            except Exception as e:
                print(e)
                print("Error with bot: " + str(bot))


class Realtime_Whisper():
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default", stream_fn=lambda x, y: print(x)):
        self.model_name = model_name
        self.device_name = device_name
        self.stream_fn = stream_fn
        self.transcript = ""
        self.bots = [KeywordBot(["hello", "hi", "hey"])]

    def stop(self):
        """stop the asr process"""
        Realtime_Whisper.exit_event.set()
        self.asr_input_queue.put("close")
        print("asr stopped")

    def start(self):
        """start the asr process"""
        manager = multiprocessing.Manager()

        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()

        # currently not used, the queue is still in for convenience...
        self.visualization_input_queue = manager.Queue()

        self.asr_process = threading.Thread(target=Realtime_Whisper.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue, self.stream_fn, self.bots))
        self.asr_process.daemon = True
        self.asr_process.start()

        time.sleep(5)  # start vad after asr model is loaded

        self.vad_process = threading.Thread(target=Realtime_Whisper.vad_process, args=(
            self.device_name, self.asr_input_queue, self.visualization_input_queue, ))
        self.vad_process.daemon = True
        self.vad_process.start()

        # Debug optional visualization
        # self.visualization_process = multiprocessing.Process(target=Realtime_Whisper.plot_stream, args=(
        #    self.visualization_input_queue,))
        # self.visualization_process = threading.Thread(target=Realtime_Whisper.plot_stream, args=(
        #     self.visualization_input_queue,))
        # self.visualization_process.daemon = True
        # self.visualization_process.start()

    def int2float(sound):
        """convert the wav pcm16 format to one suitable for silero vad"""
        _sound = np.copy(sound)  # may be not necessary
        # abs_max = np.abs(_sound).max()
        abs_max = 32767
        _sound = _sound.astype('float32')
        if abs_max > 0:
            _sound *= 1 / abs_max
        _sound = _sound.squeeze()  # depends on the use case
        return _sound

    def plot_stream(instream):
        """plot audio stream via matplotlib"""
        CHUNK = 160
        CHANNELS = 1
        RATE = 16000

        fig, ax = plt.subplots()
        x = np.arange(0, 2 * CHUNK, 2)

        line, = ax.plot(x, np.random.rand(CHUNK), 'r')
        ax.set_ylim(-20000, 20000)
        ax.set_xlim(0, CHUNK)
        fig.show()

        while True:
            data = instream.get()
            dataInt = struct.unpack(str(CHUNK) + 'h', data)
            line.set_ydata(dataInt)
            fig.canvas.draw()
            fig.canvas.flush_events()

    def vad_process(device_name, asr_input_queue, vis_input_queue):
        """voice activity detection using silero-vad"""
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils

        # not sure this is useful, but I leave it in for now...
        vad_iterator = VADIterator(model)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        FRAME_DURATION = 60
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        SPEECH_PROB_THRESHOLD = 0.2  # This probably needs a bit of tweaking

        microphones = Realtime_Whisper.list_microphones(audio)
        selected_input_device_id = Realtime_Whisper.get_input_device_id(
            device_name, microphones)
        print(microphones)

        # this should be the default mic, tweak as needed ...
        selected_input_device_id = 1

        # TODO: investigate why default input device stopped working
        stream = audio.open(
            input_device_index=selected_input_device_id,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

        # framebuffer for queue
        frames = b''
        # masterframebuffer for saving the data send to asr
        masterframes_asr = b''

        last_speech_prob = 0

        while True:
            if Realtime_Whisper.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)

            frame_tensor = torch.from_numpy(
                Realtime_Whisper.int2float(np.frombuffer(frame, dtype=np.int16)))

            speech_prob = model(frame_tensor, RATE).item()

            # turn this on for debugging and tweaking the threshold...
            # print(speech_prob)

            # accumulate frames in frame buffer if speech is detected and the total length is < 30 sec (max size of whisper chunk)
            # THIS NEEDS TO BE LOOKED AT AGAIN MAYBE A FULL 30s WHISPER CHUNK IS TOO MUCH
            if speech_prob > SPEECH_PROB_THRESHOLD and len(frames) < 480000:
                frames += frame
            # if there was speech and now there is none (i.e. an utterance has finished or the max length is exceeded, write to queue
            elif (speech_prob <= SPEECH_PROB_THRESHOLD < last_speech_prob) or (len(frames) >= 480000):
                asr_input_queue.put(frames)

                masterframes_asr += frames
                frames = b''

            last_speech_prob = speech_prob

        stream.stop_stream()
        stream.close()
        audio.terminate()
        # Open and Set the data of the WAV file
        file = wave.open(filename_orig, 'wb')
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)

        # Write and Close the File
        file.writeframes(b''.join(np.frombuffer(
            masterframes_asr, dtype=np.int16)))
        file.close()

    def queue_to_string(queue):
        output_str = ""
        while not queue.empty():
            segment = queue.get()
            for s in segment:
                output_str += s["text"]

        return output_str

    def asr_process(model_name, in_queue, output_queue, stream_fn, bots=[]):
        """transcribe using whisper"""
        model = whisper.load_model(
            model_name)  # device=cudause cuda for everything > base model

        # with current settings always excepts to 0, but left in to play around with the setting...
        temperature_increment_on_fallback = 0
        temperature = 0
        try:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
        except:
            temperature = 0

        kwargs = {}
        kwargs['language'] = 'en'
        kwargs['verbose'] = True
        kwargs['task'] = 'transcribe'
        kwargs['temperature'] = temperature
        kwargs['best_of'] = None
        kwargs['beam_size'] = None
        kwargs['patience'] = None
        kwargs['length_penalty'] = None
        kwargs['suppress_tokens'] = "-1"
        kwargs['initial_prompt'] = None
        # seems source of false Transcripts
        kwargs['condition_on_previous_text'] = False
        kwargs['fp16'] = False  # set false if using cpu
        kwargs['compression_ratio_threshold'] = None  # 2.4
        kwargs['logprob_threshold'] = None  # -1.0 #-0.5
        kwargs['no_speech_threshold'] = None  # 0.6 #0.2

        # masterframes = ''
        masterframes = b''

        while True:
            audio_file = in_queue.get()

            if audio_file == "close":
                break

            print("\nlistening to your beautiful voice\n")
            masterframes += audio_file

            audio_tensor = torch.from_numpy(Realtime_Whisper.int2float(
                np.frombuffer(audio_file, dtype=np.int16)))

            result = model.transcribe(audio_tensor, **kwargs)

            if result != "":
                output_queue.put(result["segments"])
                # TODO add segments to stream
                text = result["text"]
                if check_string(text):
                    text = ""
                stream_fn(text, "transcript")
                asyncio.run(bot_feed(bots, result["segments"], stream_fn))

        # Open and Set the data of the WAV file
        file = wave.open(filename, 'wb')
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)

        file.writeframes(b''.join(np.frombuffer(masterframes, dtype=np.int16)))
        file.close()
        full_audio = torch.from_numpy(Realtime_Whisper.int2float(
                np.frombuffer(masterframes, dtype=np.int16)))
        complete_transcript = model.transcribe(full_audio)


        stream_fn(complete_transcript['text'], 'set_output')


    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()
    



if __name__ == "__main__":
    print("Live ASR")

    # param is model size
    asr = Realtime_Whisper("small.en")

    asr.start()

    last_text = 'Start'

    try:
        while True:
            lastresult = asr.get_last_text()
            for segment in lastresult:
                print('ID: ' + str(segment['id']) + ' START: ' + str(round(segment['start'], 1)
                                                                     ) + ' END: ' + str(round(segment['end'], 1)) + ' TEXT: ' + segment['text'])
    except KeyboardInterrupt:
        asr.stop()
        exit()
