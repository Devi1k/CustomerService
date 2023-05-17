import json
import os
import time

import numpy as np
# Create your views here.
import onnxruntime as ort
import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import BertTokenizer

from Apps.ai_matter_content.dataloader import Dataloader
from Apps.ai_matter_content.utils.logger import Logger, clean_log

config = json.load(open(os.getcwd() + "/Apps/ai_matter_content/model/crosswoz_all_base.json"))
DEVICE = config['DEVICE']
data_dir = config['data_dir']
tokenizer = BertTokenizer.from_pretrained(config['model']['pretrained_weights'])
intent_vocab = json.load(open(os.getcwd() + "/Apps/ai_matter_content/model/intent_vocab.json"))

log = Logger('matter_content').getLogger()
dataloader = Dataloader(intent_vocab=intent_vocab,
                        pretrained_weights=config['model']['pretrained_weights'])
log.info(ort.__version__)
log.info(ort.get_device())
sess = ort.InferenceSession(os.getcwd() + "/Apps/ai_matter_content/model/0505matter_content.onnx",
                            providers=['CUDAExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def build_predict_text(text):
    tokens = tokenizer.tokenize(text)
    words = ['[CLS]'] + tokens + ['[SEP]']
    # print(tokens)
    word_seq_tensor = torch.zeros((1, 256), dtype=torch.long)
    word_mask_tensor = torch.zeros((1, 256), dtype=torch.long)
    indexed_tokens = tokenizer.convert_tokens_to_ids(words)
    word_seq_tensor[0, : len(words)] = torch.LongTensor(indexed_tokens)
    word_mask_tensor[0, : len(words)] = torch.LongTensor([1] * len(words))
    return word_seq_tensor, word_mask_tensor


def infer_onnx(sess, utterance):
    word_seq_tensor, word_mask_tensor = build_predict_text(
        utterance)
    input = {
        'word_seq_tensor': to_numpy(word_seq_tensor),
        'word_mask_tensor': to_numpy(word_mask_tensor),
    }
    out_name = sess.get_outputs()[0].name
    outs = sess.run([out_name], input)
    num = np.argmax(outs)
    return dataloader.id2intent[num]


warm_start_text = "我想办理公共卫生许可"
warm_start_time = time.time()
intent = infer_onnx(sess, warm_start_text)
log.info(
    'warm start text:{},intent:{},cost time:{}'.format(warm_start_text, intent, str(time.time() - warm_start_time)))


@csrf_exempt
def identify(request):
    log.info('*' * 5 + 'clean log' + '*' * 5)
    clean_log()
    log.info('-----------------------------------------------------------')
    if request.method == 'GET':
        raw_text = request.GET.get('text')
        start_time = time.time()
        intent = infer_onnx(sess, raw_text)
        log.info('text:{},intent:{},cost time:{}'.format(raw_text, intent, str(time.time() - start_time)))
        return JsonResponse({'message': 'success', 'data': intent, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
