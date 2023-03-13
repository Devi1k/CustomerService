class ResultGenerator:
    def __init__(self):
        self.resultCode = 0
        self.message = ""
        self.data = object
        self.result = dict()

    # def getSuccessResultMessage(self, message):
    #     self.resultCode = 200
    #     self.message = message

    def getSuccessResult(self, data):
        self.resultCode = 200
        self.data = data
        self.message = "SUCCESS"
        self.result['message'] = self.message
        self.result['resultCode'] = self.resultCode
        self.result['data'] = data
        return self.result

    def getFailResult(self, message):
        self.resultCode = 500
        self.message = message
        self.result['message'] = self.message
        self.result['resultCode'] = self.resultCode
        self.result['data'] = None
        return self.result

    def getNothingResult(self):
        self.result['message'] = ""
        self.result['resultCode'] = 204
        self.result['data'] = None
