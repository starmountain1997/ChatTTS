import html_text
from ebooklib import epub
import os.path as osp
import os
from loguru import logger
import sentence_spliter
from collections import OrderedDict

import sentence_spliter.spliter
import ChatTTS
import numpy as np
import torchaudio
import torch

from multiprocessing import Process, Pool


class EpubTTS:
    def __init__(self, book_name: str) -> None:
        self._root_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"root_path: {self._root_path}")

        self._book_name = book_name
        self._book_dict = None

    @property
    def book_name(self):
        return self._book_name

    @ property
    def root_path(self):
        return self._root_path

    @property
    def book_dict(self):
        return self._book_dict

    @property
    def chat(self):
        return self._chat

    @property
    def rand_spk(self):
        return self._rand_spk

    @property
    def params_infer_code(self):
        return self._params_infer_code

    @property
    def params_refine_text(self):
        return self._params_refine_text

    def read_epub(self):
        book = epub.read_epub(
            osp.join(self._root_path, f"{self._book_name}.epub"))
        items = book.get_items()
        chapters = []
        for item in items:
            contents = item.get_content()
            try:
                contents = contents.decode('utf-8')
                contents = html_text.extract_text(contents)
                if contents.find("目录") != -1 or contents.find("图书在版编目") != -1:
                    continue
                contents = contents.split('\n')
                contents = [content.strip()
                            for content in contents if content and content != "\n"]
                if contents and len(contents) > 1:
                    chapters.append(contents)
            except Exception as e:
                pass
        self._book_dict = {
            str(i): [sentence_spliter.spliter.cut_to_sentences(
                paragraph) for paragraph in chapter]
            for i, chapter in enumerate(chapters)
        }

    @staticmethod
    def chunks(l, n):
        for i in range(0, n):
            yield l[i::n]

    def _chapter_generator(self, keys):
        chat = ChatTTS.Chat()
        chat.load_models(compile=False)
        for title in keys:
            chapters = self._book_dict[title]
            wavs_list = []
            for paragraph in chapters:
                for line in paragraph:
                    wavs = chat.infer(line)
                    wavs_list = wavs_list + wavs
                    logger.info(line)
            wavs_list = np.concatenate(wavs_list, axis=1)
            torchaudio.save(osp.join(self.root_path, "output", self.book_name,
                                     f"{title}.wav"), torch.from_numpy(wavs_list), 24000)
            logger.info(f"{title} saved!")

    def _chapter_generator(self, keys):
        chat = ChatTTS.Chat()
        chat.load_models(compile=False)
        speaker = torch.load(osp.join(self.root_path, "speaker", "girl1.pth"))
        params_infer_code = {
            'prompt': '[speed_5]',
            'temperature': .3,
            'spk_emb': speaker
        }
        for title in keys:
            chapters = self._book_dict[title]
            wavs_list = []
            for paragraph in chapters:
                for line in paragraph:
                    wavs = chat.infer(line, use_decoder=True,
                                      params_infer_code=params_infer_code)
                    wavs_list = wavs_list + wavs
                    logger.info(line)
            wavs_list = np.concatenate(wavs_list, axis=1)
            torchaudio.save(
                osp.join(self.root_path, "output", self.book_name, f"{title}.wav"), torch.from_numpy(wavs_list), 24000)
            logger.info(f"{title} saved!")

    def chapter_generator(self, split: int = 2):
        if not osp.exists(osp.join(self.root_path, "output", self.book_name)):
            os.makedirs(osp.join(self.root_path, "output", self.book_name))
        split_keys = list(self.chunks(list(self._book_dict.keys()), split))
        p = Pool(split)
        for keys in split_keys:
            p.apply_async(self._chapter_generator, args=(keys,))
        p.close()
        p.join()

    def save_spk(self):
        text = "简短的测试文字"
        chat = ChatTTS.Chat()
        chat.load_models(compile=False)
        rand_spk = chat.sample_random_speaker()
        params_infer_code = {
            'prompt': '[speed_5]',
            'temperature': .3,
            'spk_emb': rand_spk,
        }
        wavs = chat.infer(text, use_decoder=True,
                          params_infer_code=params_infer_code)
        torchaudio.save(
            osp.join(self.root_path, "output", "test.wav"), torch.from_numpy(wavs[0]), 24000)
        torch.save(rand_spk, osp.join(self.root_path, "speaker", "girl4.pth"))


def main():
    et = EpubTTS("冬牧场")
    et.read_epub()
    et.chapter_generator()
    # et.save_spk()


if __name__ == '__main__':
    main()
