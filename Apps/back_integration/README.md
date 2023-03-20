# GovDiagnose

```shell
ps -ef | grep gov_service_test.py | grep -v grep | awk '{print "kill -9 "$2}' | sh
ps -ef | grep run_service.py | grep -v grep | awk '{print "kill -9 "$2}' | sh
nohup python3 run_service.py > service.out 2>&1 &
```

1. 将wb.text.model开头的三个文件放到data文件夹下

pipe_dict中的参数

0. send_pipe 主进程端
1. receive_pipe 子进程端
2. first_utterance 首句话
3. process 对应子进程
4. single_finish 单一对话结束状态
5. all_finish 整体对话结束状态
6. is_first 是否是第一句话
7. service_name 当前对话事项名称
8. last_msg 当前最后一条消息
9. dialogue_retrieval_turn 对话检索轮次
10. history 用户query历史


