import os.path

import numpy as np
import sentence_spliter.spliter
import torch
import torchaudio

import ChatTTS
from epub_generator.epub_reader import read_epub


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chapter_generator(chapters):
    chat = ChatTTS.Chat()
    chat.load_models(compile=False)  # 设置为True以获得更快速度
    rand_spk = chat.sample_random_speaker()

    params_infer_code = {
        "spk_emb": rand_spk,
        'temperature': .3,  # using custom temperature
        'top_P': 0.7,  # top P decode
        'top_K': 20,  # top K decode
    }
    params_refine_text = {
        'prompt': '[oral_2][laugh_0][break_6]'
    }
    for j, chapter in enumerate(chapters[19:]):  # 4
        title = chapter[0]
        if not os.path.exists(f"cache/{title}"):
            os.makedirs(f"cache/{title}")
        for i, lines in enumerate(chapter[1:]):
            wavs_list = []
            lines = sentence_spliter.spliter.cut_to_sentences(lines)
            for k, line in enumerate(lines):
                wavs = chat.infer(line, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
                wavs_list = wavs_list + wavs
                print(line)
            wavs_list = np.concatenate(wavs_list, axis=1)
            np.save(f"cache/{title}/{i}.npy", wavs_list)


def convert_npy_to_wav():
    article_list = os.listdir("cache")
    for article in article_list:
        chapters = os.listdir(f"cache/{article}")
        wavs_list = []
        for chapter in chapters:
            with open(f"cache/{article}/{chapter}", 'rb') as f:
                wavs = np.load(f)
            wavs_list.append(wavs)
        wavs_list = np.concatenate(wavs_list, axis=1)
        torchaudio.save(f"output/{article}.wav", torch.from_numpy(wavs_list), 24000)


if __name__ == '__main__':
    chapters = read_epub("冬牧场.epub")
    chapter_generator(chapters)
    convert_npy_to_wav()
