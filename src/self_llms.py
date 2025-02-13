# -*- coding: utf-8 -*-
# @Time    : 2025/2/13 23:47
# @Author  : yblir
# @File    : self_llms.py
# explain  : 自定义推理和嵌入大模型
# =======================================================
import os
from typing import Any, List
from llama_index.llms import vllm
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_serializer,
)

from llama_index.core.llms.callbacks import llm_completion_callback


class VllmLLM(CustomLLM):
    """自定义大模型, 模型是huggingface格式"""
    # self.vllm_model字段必须先在此声明才能用
    vllm_model: Any = Field(default=None, description="VLLM 模型实例")
    def __init__(self, model_path: str):
        super().__init__()
        if not os.path.exists(model_path):
            raise ValueError("模型路径不存在")

        self.vllm_model = vllm.Vllm(model_path)

    # 实现metadata 接口
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
                # model_name=Field(default='vllm_llm')
                model_name='vllm_model'
        )

    # 实现complete 接口
    @llm_completion_callback()
    def complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponse:
        response = self.vllm_model.complete(prompt, **kwargs)
        return CompletionResponse(
                text=response.text
        )

    # 实现stream_complete 接口
    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        model_response = self.vllm_model.complete(prompt, **kwargs)

        for token in model_response.text:
            response += token
            yield CompletionResponse(text=response, delta=token)


class SelfEmbeddingLLM(BaseEmbedding):
    # 导入自己的嵌入模型提供的模块，实现embed 方法
    def __init__(
            self,
            model_name: str = "MyEmbeddingModel",
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # 构造一个模型调用对象（模拟）
        self._model = MyModel(model_name)

    # 生成向量（模拟）
    def get_text_embedding(self, text: str) -> List[float]:
        embedding = self._model.embed(text)
        return embedding

    # 批量生成向量（模拟）
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.embed(
                [text for text in texts]
        )
        return embeddings


if __name__ == '__main__':
    # llm = VllmLLM(r'E:\PyCharm\PreTrainModel\qwen2_7b_instruct_awq_int4')
    llm=VllmLLM('/mnt/e/PyCharm/PreTrainModel/qwen2_7b_instruct_awq_int4')
    res = llm.complete('你好,介绍下你自己')
    print(res.text)
