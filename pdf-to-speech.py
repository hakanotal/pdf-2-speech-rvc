# !pip install -qq noisereduce
# !pip install -qq infer-rvc-python
# !pip uninstall fairseq -y
# !pip install git+https://github.com/Tps-F/fairseq.git@main


from unstructured.partition.pdf import partition_pdf
from infer_rvc_python import BaseLoader
from datasets import load_dataset
from transformers import pipeline
from pydub import AudioSegment
import noisereduce as nr
import scipy.io.wavfile
from tqdm import tqdm
import unstructured
import numpy as np
import traceback
import shutil
import torch
import re
import os

np.float = float    

################################################################################################################################


input_pdf_file = "test.pdf" # pdf file
output_dir = "audio_files" # tts audio files
final_dir = "final_nr" # final audio files after noise reduction


# https://huggingface.co/Its7up/Disco-Elysium-Narrator-RVC-v2
model_path = './models/de_narrator.pth'
index_path = './models/logs/de_narrator/added_IVF256_Flat_nprobe_1_de_narrator_v2.index'


################################################################################################################################


# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)


# Create the final directory if it doesn't exist
if not os.path.exists(final_dir):
    os.makedirs(final_dir)
else:
    shutil.rmtree(final_dir)
    os.makedirs(final_dir)


################################################################################################################################

elements = partition_pdf(input_pdf_file)

# Filter texts
narrative_texts = []

for element in elements:
    if element.category == 'NarrativeText':
        narrative_texts.append(element.text)
    elif element.category == 'Text':
        narrative_texts.append(element.text)
    elif element.category == 'Title':
        print(element.text)
        narrative_texts.append(str(element.text+'.'))
        
# Combine narrative texts
combined_text = ' '.join(narrative_texts)
sentences = re.split(r'[.!?:·] \s*', combined_text)
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

print('Number of sentences:', len(sentences))

################################################################################################################################

# Initialize the Hugging Face TTS pipeline
tts = pipeline("text-to-speech", model='microsoft/speecht5_tts')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speaker_embeddings = embeddings_dataset.filter(lambda example: example["filename"] == "cmu_us_bdl_arctic-wav-arctic_a0001")[0]['xvector']
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

# Process each page of the PDF file
for i, text in enumerate(tqdm(sentences)):

    try:
        # Convert the chunk to audio using Hugging Face TTS
        audio_data = tts(text, forward_params={"speaker_embeddings": speaker_embeddings})

        # Extract the audio signal and sampling rate
        audio_signal = audio_data["audio"]
        sampling_rate = audio_data["sampling_rate"]

        # Save the audio file
        audio_file = f"{output_dir}/sentence_{i+1}.wav"
        scipy.io.wavfile.write(audio_file, sampling_rate, audio_signal)
        # print(f"Saved audio file: {audio_file}")

    except Exception as e:
        print(f"Error converting text to audio: {str(e)}")


################################################################################################################################

converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)

converter.apply_conf(
        tag="de_narrator",
        file_model=model_path,
        pitch_algo="rmvpe+",
        file_index=index_path,
        pitch_lvl=0,
        index_influence=0.75,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.5
    )

def apply_rvc(audio_files):

    speakers_list = ["de_narrator"]

    result = converter(
        audio_files,
        speakers_list,
        overwrite=False,
        parallel_workers=8
    )

    return result

output_files = os.listdir(output_dir)
output_files.sort(key=lambda x: int(re.sub(r'sentence_(\d+)\.wav', r'\1', x)))
audio_files = [f"{output_dir}/{f}" for f in output_files]

apply_rvc(audio_files)

################################################################################################################################

def apply_noisereduce(audio_list):

    result = []
    for audio_path in tqdm(audio_list):
        out_path = f"{final_dir}/{os.path.splitext(audio_path)[0].split('/')[-1]}.wav"

        try:
            # Load audio file
            audio = AudioSegment.from_file(f'{os.path.splitext(audio_path)[0]}_edited.wav')

            # Convert audio to numpy array
            samples = np.array(audio.get_array_of_samples())

            # Reduce noise
            reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate, prop_decrease=0.6)

            # Convert reduced noise signal back to audio
            reduced_audio = AudioSegment(
                reduced_noise.tobytes(), 
                frame_rate=audio.frame_rate, 
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            # Save reduced audio to file
            reduced_audio.export(out_path, format="wav")
            result.append(out_path)

        except Exception as e:
            traceback.print_exc()
            print(f"Error noisereduce: {str(e)}")
            result.append(audio_path)

    return result



apply_noisereduce(audio_files)


################################################################################################################################


def combine_audio_files(final_dir):

    audio_files = os.listdir(final_dir)
    audio_files.sort(key=lambda x: int(re.sub(r'sentence_(\d+)\.wav', r'\1', x)))
    audio_files = [f"{final_dir}/{f}" for f in audio_files]

    combined_audio = AudioSegment.empty()

    for audio_file in tqdm(audio_files):
        audio_segment = AudioSegment.from_file(audio_file)
        slowed_audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": int(audio_segment.frame_rate * 0.9)})
        combined_audio += slowed_audio_segment
        combined_audio += AudioSegment.silent(duration=700)  # add silence

    combined_audio.export(f"combined.wav", format="wav")

    return combined_audio

combine_audio_files(final_dir)