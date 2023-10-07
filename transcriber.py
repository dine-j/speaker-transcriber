#!/usr/bin/env python3
"""
Usage: python3 transcriber.py -a audio_file
"""
import getopt
import re
import sys
import torch
import whisper
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from whisper.utils import format_timestamp
import numpy
from pyannote.core import Segment
from pyannote.audio import Audio
import wave
import contextlib


def transcribe(audio_file):
    model_name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    print("Transcribing text")
    model = whisper.load_model(model_name, device=torch.device('cuda'))
    result = model.transcribe(audio_file)
    # format_result('transcription.txt', result["text"])
    format_result_speaker(audio_file, 'transcription.txt', result["segments"])


def format_result(file_name, text):
    """Put a newline character after each sentence."""
    format_text = re.sub('\.', '.\n', text)
    with open(file_name, 'a', encoding="utf-8") as file:
        print("Writing transcription to text file")
        file.write(format_text)


def get_duration(audio_file):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def segment_embedding(audio_file, segment):
    audio = Audio()
    clip = Segment(segment["start"], min(get_duration(audio_file), segment["end"]))
    waveform, sample_rate = audio.crop(audio_file, clip)
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda"))
    return embedding_model(waveform[None])


def format_result_speaker(audio_file, file_name, segments):
    """Put a newline character after each sentence."""
    text = ""
    # embeddings = numpy.zeros(shape=(len(segments), 192))
    # for i, segment in enumerate(segments):
    #     embeddings[i] = segment_embedding(audio_file, segment)
    #
    # embeddings = numpy.nan_to_num(embeddings)
    # clustering = AgglomerativeClustering(2).fit(embeddings)
    # labels = clustering.labels_
    # for i in range(len(segments)):
    #     segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    for segment in segments:
        text += "[" + format_timestamp(segment["start"]) + "] SPEAKER:\n"
        text += segment['text'] + "\n\n"
    with open(file_name, 'a', encoding="utf-8") as file:
        print("Writing transcription to text file")
        file.write(text)


def main():
    audio = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "a:", ["audio="])
    except:
        print("Usage: python3 transcriber.py -a <audio_file>")
    for opt, arg in opts:
        if opt in ['-a', '--audio']:
            audio = arg

    transcribe(audio)  # Get audio transcription and translation if needed


if __name__ == "__main__":
    main()
