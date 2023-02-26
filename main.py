import pathlib as pl
import whisper
from pytube import YouTube

model = whisper.load_model("base", device='cuda')

def get_text(url):
    yt = YouTube(url)
    if yt.length < 5400:
        audio_stream = yt.streams.filter(only_audio=True).first()
        out_file = audio_stream.download(output_path='.')
        result = model.transcribe(out_file)
        pl.Path(out_file).unlink(missing_ok=True)
        return result['text'].strip()
    

if __name__ == '__main__':
    import argparse

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-u", "--url")
    args = argParser.parse_args()
    text = get_text(args.url)
    print(text)