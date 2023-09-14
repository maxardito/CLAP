from src.models.clap import CLAP
from src.CLAPWrapper import CLAPWrapper

clap_model = CLAPWrapper(model_fp="../CLAP.pth", use_cuda=True)

text_prompts = [
    "drums", "hip hop", "uk garage", "bass", "orchestra", "acoustic",
    "electro", "70s", "the", "help", "republican"
]
audio_file = "../UK2S_130_Drum_Goin_Full.wav"

text_embeddings = clap_model.get_text_embeddings(text_prompts)
audio_embeddings = clap_model.get_audio_embeddings([audio_file], 1)

similarity = clap_model.compute_similarity(text_embeddings=text_embeddings,
                                           audio_embeddings=audio_embeddings)

similarity = similarity.cpu().detach().numpy()[0]

for prompt, sim in list(zip(text_prompts, similarity)):
    print(f"{prompt}: {sim}")
