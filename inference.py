import os
import json
import tqdm
import soundfile as sf

AUDIO_FILETYPES = tuple(["." + filetype.lower() for filetype in list(sf._formats.keys())] + [".mp3"])

from src.models.clap import CLAP
from src.CLAPWrapper import CLAPWrapper

CLAP_MODEL = "../CLAP.pth"
KEYWORDS_FILE = "keywords.txt"
AUDIO_DIR = "../audio/points005"

clap_model = CLAPWrapper(model_fp=CLAP_MODEL, use_cuda=True)

prompts = open(KEYWORDS_FILE).read().splitlines()
prompt_batch_size = 1000
num_prompts = len(prompts)
# num_prompts = prompt_batch_size


## FOR each file:
# audio_file = os.path.join(AUDIO_DIR, "06. LTJ Bukem - Music.mp3")

for root, subdirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if os.path.splitext(file)[-1].lower() in AUDIO_FILETYPES:
            file = os.path.join(root, file)
            basename = os.path.basename(file)
            basename = os.path.splitext(basename)[0]
            audio_embeddings = clap_model.get_audio_embeddings([file], 1)

            results = []

            for prompt_batch_start in tqdm.tqdm(range(0, num_prompts, prompt_batch_size)):
                # get a batch of prompts
                if prompt_batch_start + prompt_batch_size < num_prompts:
                    prompt_batch = prompts[prompt_batch_start:prompt_batch_start+prompt_batch_size]
                else:
                    prompt_batch = prompts[prompt_batch_start:num_prompts]

                text_embeddings = clap_model.get_text_embeddings(prompt_batch)

                similarity = clap_model.compute_similarity(text_embeddings=text_embeddings,
                                                        audio_embeddings=audio_embeddings)

                similarity = similarity.cpu().detach().numpy()[0]

                batch_results = [
                    {
                        "prompt": prompt, 
                        "value": str(sim)
                    } 
                    for prompt, sim in list(zip(prompts, similarity))
                    ]

                results = results + batch_results

            with open(f'{basename}.json', 'w') as f:
                json.dump(results, f, indent="\t")