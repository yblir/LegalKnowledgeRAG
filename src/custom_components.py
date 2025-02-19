# -*- coding: utf-8 -*-
# @Time    : 2025/2/13 23:47
# @Author  : yblir
# @File    : custom_components.py
# explain  : 自定义推理和嵌入大模型
# =======================================================
import os
from typing import Any, List, Optional, Sequence

from llama_index.core.base.embeddings.base import Embedding
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
    # ConfigDict,
    # field_serializer,
)
from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.llms import vllm
from llama_index.llms.ollama import Ollama
from pydantic import Field

from llama_index.core.llms.callbacks import llm_completion_callback


# 自定义响应器
class CustomSynthesizer(BaseSynthesizer):
    my_prompt = (
        "根据以下上下文信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "使用中文回答以下问题\n "
        "问题: {query_str}\n"
        "答案: "
    )

    def __init__(
            self,
            llm: Optional[LLMPredictorType] = None,
    ) -> None:
        super().__init__(llm=llm)
        self._input_prompt = PromptTemplate(CustomSynthesizer.my_prompt)

    # 必须实现的接口
    def _get_prompts(self) -> PromptDictType:
        return self._input_prompt.text

    # 必须实现的接口, 更新提示词
    def _update_prompts(self, prompts: PromptDictType) -> None:
        self._input_prompt = PromptTemplate(prompts.text)

    # 生成响应的接口
    def get_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join(n for n in text_chunks)
        # 此处可以自定义任何响应逻辑
        response = self._llm.predict(
                self._input_prompt,
                query_str=query_str,
                context_str=context_str,
                **response_kwargs,
        )
        return response

    # 响应接口的异步版本
    async def aget_response(
            self,
            query_str: str,
            text_chunks: Sequence[str],
            **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        context_str = "\n\n".join(n for n in text_chunks)
        response = await self._llm.apredict(
                self._input_prompt,
                query_str=query_str,
                context_str=context_str,
                **response_kwargs,
        )
        return response


class CustomVllmLLM(CustomLLM):
    """自定义大模型, 模型是huggingface格式"""
    # self.vllm_model字段必须先在此声明才能用
    vllm_model: vllm.Vllm = Field(default=None, description="VLLM 模型实例")

    def __init__(self, model_path: str):
        super().__init__()
        if not os.path.exists(model_path):
            raise ValueError("模型路径不存在")

        self.vllm_model = vllm.Vllm(model_path)

    # 实现metadata 接口
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
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


# 自定义嵌入模型
class CustomEmbeddingLLM(BaseEmbedding):
    vllm_embedding_model: Any = Field(default=None, description="VLLM 模型实例")

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
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.embed(
                [text for text in texts]
        )
        return embeddings

    def _get_text_embedding(self, text: str) -> Embedding:
        pass

    async def _aget_query_embedding(self, query: str) -> Embedding:
        pass

    def _get_query_embedding(self, query: str) -> Embedding:
        pass


if __name__ == '__main__':
    # llm = CustomVllmLLM(r'E:\PyCharm\PreTrainModel\qwen2_7b_instruct_awq_int4')
    llm = CustomVllmLLM('/mnt/e/PyCharm/PreTrainModel/qwen2_7b_instruct_awq_int4')
    res = llm.complete('你好,介绍下你自己')
    print(res.text)
