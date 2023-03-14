# speaker-transcriber

When you're talking for hours and want a detailed recap.

## transcriber

Simple python program to test out [Whisper](https://github.com/openai/whisper) (from OpenAI).

Usage: `python3 transcriber.py -a audio_file`

## speaker-transcriber

If you want speakers, this combines an existing trained model ([here](https://huggingface.co/pyannote/speaker-diarization) - notice you'll need an account and a token to use it!) that identifies speakers.
The program splits the audio file per speaker, then run Whisper on all the audio bits.

Usage: `python3 speaker-transcriber.py -a audio_file.mp3 -d True -s 4 -t <token>`