# -*- coding: utf-8 -*-
# @Time    : 2025/2/16 19:24
# @Author  : yblir
# @File    : custom_query_engine.py
# explain  : 
# =======================================================
from typing import Optional, Sequence, Any

from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.llms import vllm
from llama_index.llms.ollama import Ollama
from pydantic import Field

from custom_components import CustomVllmLLM, CustomSynthesizer


# 单次查询引擎
# 对chat_engine =vector_index.as_query_engine()的自定义操作
class OnceQueryEngine(CustomQueryEngine):
    # 此处直接使用大模型组件，而不是响应生成器
    # llm: Ollama = Field(default=None, description="llm")
    llm: vllm.Vllm = Field(default=None, description="llm")
    retriever: BaseRetriever = Field(default=None, description="retriever")
    qa_prompt: PromptTemplate = Field(default=None, description="提示词")
    synthesizer: CustomSynthesizer = Field(default=None, description="自定义响应器")

    qa_prompt = PromptTemplate(
            "根据以下上下文回答输入问题：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "回答以下问题，不要编造\n"
            "我的问题: {query_str}\n"
            "答案: "
    )

    def __init__(self, retriever: BaseRetriever, llm: CustomVllmLLM):
        super().__init__()
        self.retriever = retriever
        # self.llm = llm
        self.synthesizer = CustomSynthesizer(llm=llm)

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        # 用检索出的Node 构造上下文
        context_str = "\n\n".join([n.node.get_content() for n in nodes])

        # 用上下文与查询问题组装Prompt,然后调用大模型组件响应生成
        # response = self.llm.complete(
        #         OnceQueryEngine.qa_prompt.format(
        #               context_str=context_str, query_str=query_str
        #         )
        # )

        # 使用自定义响应器完成响应生成
        response = self.synthesizer.get_response(
                query_str=query_str,
                text_chunks=context_str
        )
        return str(response)


# 对话查询引擎
# 对chat_engine = vector_index.as_chat_engine(chat_mode="condense_question")自定义操作
class ChatQueryEngine:
    custom_prompt = PromptTemplate(
            """
            请根据以下的历史对话记录和新的输入问题，重写一个新的问题，使其能够捕捉对话中的
            所有相关上下文。
            <Chat History>
            {chat_history}
            <Follow Up Message>
            {question}
            <Standalone question>
            """
    )
    # 历史对话记录
    custom_chat_history = [
        ChatMessage(
                role=MessageRole.USER,
                content="我们来讨论一些有关法律知识的问题",
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="好的"),
    ]

    def __init__(self, retriever: BaseRetriever, llm: vllm.Vllm):
        super().__init__()
        self.once_query_engine = OnceQueryEngine(retriever, llm)
        # 这种对话模式在理解历史对话记录的基础上将当前输入的问题重写成一
        # 个独立的、具备完整语义的问题，然后通过查询引擎获得答案
        self.custom_chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=self.once_query_engine,
                # 对话引擎基于查询引擎构造
                condense_question_prompt=ChatQueryEngine.custom_prompt,  # 设置重写问题的 Prompt 模板
                chat_history=ChatQueryEngine.custom_chat_history,
                # 携带历史对话记录
                verbose=True,
        )
        self.custom_chat_engine.chat_repl()


if __name__ == '__main__':
    vllm_model = vllm.Vllm('/media/xk/D6B8A862B8A8433B/data/qwen2_05b')
    # vllm_model = CustomVllmLLM('/media/xk/D6B8A862B8A8433B/data/qwen2_05b')
    s = CustomSynthesizer(vllm_model)
    res = s.get_prompts()
    print(res)
    s.update_prompts(PromptTemplate('fdsffds').text)
    print(s.get_prompts())
    chat_engine = ChatQueryEngine('a', 'b').custom_chat_engine
    chat_engine.chat_repl()
