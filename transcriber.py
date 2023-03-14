#!/usr/bin/env python3
"""
Usage: python3 transcriber.py -a audio_file
"""
import getopt
import re
import sys
import torch
import whisper


def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device


def transcribe(audio_file):
    model_name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    print("Transcribing text")
    model = whisper.load_model(model_name, device=check_device())
    result = model.transcribe(audio_file)
    format_result('transcription.txt', result["text"])


def format_result(file_name, text):
    """Put a newline character after each sentence."""
    format_text = re.sub('\.', '.\n', text)
    with open(file_name, 'a', encoding="utf-8") as file:
        print("Writing transcription to text file")
        file.write(format_text)


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
