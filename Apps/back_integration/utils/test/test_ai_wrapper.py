from unittest import TestCase

from ..ai_wrapper import get_faq_from_service, get_recommend,get_related_title


class Test(TestCase):
    def test_get_faq_from_service(self):
        # score, answer, service = get_faq_from_service("需要第三方吗",
        #                                               "固定资产投资项目合理用能审查")  # add assertion here
        sentence = "我的普通话考试等级是三级甲等，能不能办理教师资格证"

        service = "教师资格的认定"

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
