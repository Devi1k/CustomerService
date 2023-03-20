from unittest import TestCase

from ..ai_wrapper import *


class Test(TestCase):
    def test_get_faq_from_service(self):
        # score, answer, service = get_faq_from_service("需要第三方吗",
        #                                               "固定资产投资项目合理用能审查")  # add assertion here
        sentence = "在哪里办理"

        service = "申办法定代表人证明公证咨询"

        score, answer, service = get_faq_from_service(sentence,
                                                      service, history=[])  # add assertion here

        print(score, answer)

    def test_get_recommend(self):
        service_name = "户外广告及临时悬挂、设置标语或者宣传品许可-户外广告设施许可（不含公交候车亭附属广告及公交车体广告设施）（市级权限委托市内六区实施）"
        history = ['我想挂个牌匾']
        res = get_recommend(service_name, history)
        print(res)

    def test_get_related_title(self):
        text = "新车上牌办理什么业务"
        res = get_related_title(text)
        print(res)
        # self.fail()

    def test_get_faq(self):
        text = "在哪里办理"
        similar_score, answer, service = get_faq(text)
        print(similar_score, answer, service)