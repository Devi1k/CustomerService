import json
import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler
from time import strftime, gmtime

import torch
from django.http import JsonResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

# from convlab2.util.file_util import cached_path
from .nlp_cls.IntentNLU import NLU
from .nlp_cls.dataloader import Dataloader
from .nlp_cls.jointBERT import JointBERT
from .nlp_cls.postprocess import recover_intent
from .util.logger import Logger

log = Logger('intent').getLogger()


class BERTNLU(NLU):
    def __init__(self, config_file='crosswoz_all.json', model_file='bert_crosswoz.zip'):
        # assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'ai_intent/config/{}'.format(config_file))
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        # if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
        #     preprocess()

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        log.info('intent num:{}'.format(len(intent_vocab)))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        log.info('Load from:{}'.format( best_model_path))
        model = JointBERT(config['model'], DEVICE, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        log.info("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        # ori_tag_seq = ['O'] * len(ori_word_seq)
        context_size = 1
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-context_size:]))
        intents = []
        da = {}

        word_seq, new2ori = ori_word_seq, None
        batch_data = [[ori_word_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, intent_tensor, word_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        context_seq_tensor, context_mask_tensor = None, None
        intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                           context_seq_tensor=context_seq_tensor,
                                           context_mask_tensor=context_mask_tensor)[0]
        intent_logits = intent_logits.detach().cpu().numpy()
        intent = recover_intent(self.dataloader, intent_logits[0],
                                batch_data[0][0], batch_data[0][-4])
        return intent[0][0]


log.info('model loading')
nlu = BERTNLU(config_file='crosswoz_all.json')
log.info('warming up')
start_time = time.time()
log.info('text:人才绿卡A卡的办理条件,intent:{},cost_time:{}'.format(nlu.predict(
    '人才绿卡A卡的办理条件'),
    str(time.time()-start_time)))
log.info('warm up finish. Waiting for messages')


def clean_log():
    path = os.getcwd() + '/Apps/ai_intent/log/'
    for i in os.listdir(path):
        if len(i) < 7:
            continue
        file_path = path + i  # 生成日志文件的路径
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        # 获取日志的年月，和今天的年月
        today_m = int(timestamp[4:6])  # 今天的月份
        file_m = int(i[12:14])  # 日志的月份
        today_y = int(timestamp[0:4])  # 今天的年份
        file_y = int(i[7:11])  # 日志的年份
        # 对上个月的日志进行清理，即删除。
        # print(file_path)
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)


@csrf_exempt
def intent_cls(request):
    # print(request.method)
    log.info('*' * 5 + 'clean log' + '*' * 5)
    clean_log()
    log.info('-----------------------------------------------------------')
    if request.method == 'GET':
        raw_text = request.GET.get('text')
        intent = nlu.predict(raw_text)
        log.info('text:{},intent:{}'.format(raw_text, intent))
        return JsonResponse({'message': 'success', 'data': intent, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
