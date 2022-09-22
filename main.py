import io
import os

import torch
import pandas as pd
import urllib
import tarfile
import whisper

from scipy.io import wavfile
from tqdm import tqdm

class Fleurs(torch.utils.data.Dataset):
    """
    A simple class to wrap Fleurs and subsample a portion of the dataset as needed.
    """

    def __init__(self, lang, device, split="test", subsample_rate=1):
        url = f"https://storage.googleapis.com/xtreme_translations/FLEURS102/{lang}.tar.gz"
        tar_path = os.path.expanduser(f"~/.cache/fleurs/{lang}.tgz")
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)

        if not os.path.exists(tar_path):
            with urllib.request.urlopen(url) as source, open(tar_path, "wb") as output:
                with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                          unit_divisor=1024) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))

        labels = {}
        all_audio = {}
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(f"{split}.tsv"):
                    labels = pd.read_table(tar.extractfile(member), names=(
                        "id", "file_name", "raw_transcription", "transcription", "_", "num_samples", "gender"))

                if f"/{split}/" in name and name.endswith(".wav"):
                    audio_bytes = tar.extractfile(member).read()
                    all_audio[os.path.basename(name)] = wavfile.read(io.BytesIO(audio_bytes))[1]

        self.labels = labels.to_dict("records")[::subsample_rate]
        self.all_audio = all_audio
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        record = self.labels[item]
        audio = torch.from_numpy(self.all_audio[record["file_name"]].copy())
        text = record["transcription"]

        return (audio, text)

# Display options for pandas dataset
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000
# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Set device to {device}")
# Set language (korean)
language_google = "ko_kr"
language_whisper = "korean"
print(f"Set language to {language_whisper}")

# Create dataset object, selecting only 10 examples for brevity
dataset = Fleurs(language_google, subsample_rate=1, device=device)
dataset = torch.utils.data.random_split(dataset, [10, len(dataset)-10])[0]
print(f"Created dataset")

# Load tiny Whisper model
model_size = "tiny"
model = whisper.load_model(model_size)
print(f"Loaded {model_size} Whisper model")

# Set options
options = dict(language=language_whisper, beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)
translate_options = dict(task="translate", **options)

# Run inference
references = []
transcriptions = []
translations = []

for idx, (audio, text) in enumerate(tqdm(dataset)):
    print(f"\nProcessing datum number {idx+1}")
    transcription = model.transcribe(audio, **transcribe_options)["text"]
    translation = model.transcribe(audio, **translate_options)["text"]

    transcriptions.append(transcription)
    translations.append(translation)
    references.append(text)

# Create dataframe from results and save the data
data = pd.DataFrame(dict(reference=references, transcription=transcriptions, translation=translations))
print(data)
data.to_csv("results.csv")
