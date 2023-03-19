from unittest import TestCase

from utils.logger import clean_log


class Test(TestCase):
    def test_clean_log(self):
        clean_log()
        # self.fail()
