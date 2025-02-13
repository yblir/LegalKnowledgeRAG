# -*- coding: utf-8 -*-
# @Time    : 2025/2/7 22:14
# @Author  : yblir
# @File    : tt.py
# explain  : 
# =======================================================
import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
# from llama_index.llms.ollama import Ollama
# from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms import vllm
os.environ["OPENAI_BASE_URL"] = 'https://xiaoai.plus/v1'
os.environ["OPENAI_API_KEY"] = 'O9DxD7G33V3EsuYcNwAIHsiTbZqc8lkTFlGeYDmvqcXD6n6X'

llm = vllm.Vllm(model='/mnt/e/PyCharm/PreTrainModel/qwen2_7b_instruct_awq_int4')
# 测试complete 接口
print('1')
# llm = OpenAI(model='gpt-3.5-turbo')
print('2')
resp = llm.complete("白居易是")
print('3')
print(resp)
print('-------------------------------')
# 测试chat 接口
messages = [
    ChatMessage(
            role="system", content="你是一个聪明的AI 助手"
    ),
    ChatMessage(role="user", content="你叫什么名字？"),
]
resp = llm.chat(messages)
print(resp)
