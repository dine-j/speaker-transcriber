#!/usr/bin/env python3
"""
Usage: python3 speaker-transcriber.py -a audio_file.mp3 -d True -s 4 -t <token>
"""
import getopt
import re
import sys
import torch
import whisper
import os
import datetime

from pyannote.audio import Pipeline
from pydub import AudioSegment

pattern = r"(\d*\d\:\d\d\:\d\d[\.\d+]*)\-(SPEAKER\_\d\d)"


def get_result(model, audio_file):
    matches = re.finditer(pattern, audio_file)
    hour = ""
    speaker = ""
    for m in matches:
        hour = m.group(1)
        speaker = m.group(2)

    text = "[" + hour + "] " + speaker + ":\n"
    result = model.transcribe(audio_file)
    text += result["text"]
    return text


def get_time(file_name):
    matches = re.finditer(pattern, file_name)
    time = ""
    for m in matches:
        time = m.group(1)
    return time


def diarize(auth_token, audio, audio_format, n_speakers):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    pipeline.to('cuda')
    padding_in_seconds = 2
    prep_audio_filename = 'input_prep.wav'
    prep_audio = prepare_audio_for_diarization(audio, audio_format, padding_in_seconds, prep_audio_filename)

    print("Diarizing start")
    if n_speakers == 0:
        diarization = pipeline(prep_audio_filename)
    else:
        diarization = pipeline(prep_audio_filename, num_speakers=n_speakers)
    print("Diarizing done; splitting into files")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        split = prep_audio[turn.start * 1000:turn.end * 1000]
        split.export('splits/' + str(datetime.timedelta(seconds=turn.start-padding_in_seconds)) + '-' + str(speaker) + '.wav',
                     format='wav')
    print("Split done")


def prepare_audio_for_diarization(audio, audio_format, padding_in_seconds, prep_audio_filename):
    sound = AudioSegment.from_file(audio, audio_format)
    spacer = AudioSegment.silent(duration=padding_in_seconds * 1000)
    sound = spacer.append(sound, crossfade=0)
    sound.export(prep_audio_filename, format='wav')
    return sound


def transcribe():
    text = ""
    listdir = os.listdir("splits/")
    filteredlist = filter(lambda x: 'wav' in x, listdir)
    sortedlist = sorted(filteredlist, key=lambda x: get_time(x))
    model = whisper.load_model("small.en", device=torch.device('cuda'))
    for filename in sortedlist:
        text += get_result(model, "splits/" + filename)
        text += "\n\n"
    return text


def main():
    audio = None
    should_diarize = False
    n_speakers = 0
    auth_token = ""
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "a:d:s:t:", ["audio=", "diarize=", "n_speakers=", "token="])
    except:
        print("Usage: python3 transcriber.py -a <audio_file>")
    for opt, arg in opts:
        if opt in ['-a', '--audio']:
            audio = arg
        elif opt in ['-d', '--diarize']:
            should_diarize = arg == "True"
        elif opt in ['-s', '--n_speakers']:
            n_speakers = int(arg)
        elif opt in ['-t', '--token']:
            auth_token = arg

    audio_info = audio.split(".")
    filename = audio_info[0]
    extension = audio_info[1]

    if should_diarize:
        diarize(auth_token=auth_token, audio=audio, audio_format=extension, n_speakers=n_speakers)

    print("Transcription start")
    text = transcribe()
    print("Transcription done")

    print("Writing to file")
    with open(filename + ".txt", 'a', encoding="utf-8") as file:
        file.write(text)


if __name__ == "__main__":
    main()
