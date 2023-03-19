import time
from unittest import TestCase

import gensim
import synonyms

from utils.word_match import is_multi_round, compare


class Test(TestCase):
    def test_is_multi_round(self):
        print(is_multi_round("环评的条件",
                             "固定资产投资项目合理用能审查"))

    def test_compare(self):
        model1 = gensim.models.Word2Vec.load('../../data/wb.text.model')
        model2 = gensim.models.KeyedVectors.load_word2vec_format('../../data/words.vector.gz', binary=True)

        start_time = time.time()
        print(compare('材料', '手续', model1))
        end_time = time.time()
        print(end_time - start_time)

        word_vector = model1.wv
        del model1
        word_vector.init_sims(replace=True)

        start_time = time.time()
        print(compare('材料', '手续', word_vector))
        end_time = time.time()
        print(end_time - start_time)

        start_time = time.time()
        print(compare('材料', '手续', model2))
        end_time = time.time()
        print(end_time - start_time)

        start_time = time.time()
        print(synonyms.compare('材料', '手续'))
        end_time = time.time()
        print(end_time - start_time)
