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

from whisper.utils import format_timestamp
from pyannote.audio import Pipeline
from pydub import AudioSegment

pattern = r"(\d*\d\:\d\d\:\d\d[\.\d+]*)\-(SPEAKER\_\d\d)"
padding_in_seconds = 2
prep_audio_filename = 'input_prep.wav'


def get_time(file_name):
    matches = re.finditer(pattern, file_name)
    time = ""
    for m in matches:
        time = m.group(1)
    return time


def diarize(auth_token, n_speakers):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    pipeline.to('cuda')

    print("Diarizing start")
    if n_speakers == 0:
        diarization = pipeline(prep_audio_filename)
    else:
        diarization = pipeline(prep_audio_filename, num_speakers=n_speakers)
    return diarization


def prepare_audio_for_diarization(audio, audio_format):
    sound = AudioSegment.from_file(audio, audio_format)
    spacer = AudioSegment.silent(duration=padding_in_seconds * 1000)
    sound = spacer.append(sound, crossfade=0)
    sound.export(prep_audio_filename, format='wav')
    print("Input sound created")
    return sound


def transcribe(audio_file):
    print("Transcribing text")
    model = whisper.load_model("small.en", device=torch.device('cuda'))
    return model.transcribe(audio_file)


def format_result_speaker(file_name, segments, speakers):
    """Put a newline character after each sentence."""
    text = ""
    prev_speaker = ""

    for segment in segments:
        speaker = "SPEAKER"
        if segment['id'] in speakers.keys():
            speaker = speakers[segment['id']]

        if speaker != prev_speaker or (speaker == prev_speaker and speaker == "SPEAKER"):
            text += "[" + format_timestamp(segment["start"]) + "] " + speaker + ":\n"
            prev_speaker = speaker
        text += segment['text'] + "\n\n"
    with open(file_name, 'a', encoding="utf-8") as file:
        print("Writing transcription to text file")
        file.write(text)


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
        prepare_audio_for_diarization(audio, extension)
        diarization = diarize(auth_token=auth_token, n_speakers=n_speakers)

    print("Transcription start")
    transcription = transcribe(prep_audio_filename)
    print("Transcription done")

    print("Matching speakers with segments")
    diarization_speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_speakers.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    speakers = {}
    for segment in transcription["segments"]:
        max_overlap = 0
        for ds in diarization_speakers:
            overlap = min(ds['end'], segment['end']) - max(ds['start'], segment['start'])
            if overlap >= max_overlap:
                max_overlap = overlap
                speakers[segment['id']] = ds['speaker']

    format_result_speaker(filename + ".txt", transcription["segments"], speakers)
    os.remove(prep_audio_filename)


if __name__ == "__main__":
    main()
