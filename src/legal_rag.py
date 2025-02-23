# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 下午2:23
# @Author  : yblir
# @File    : legal_rag.py
# explain  : 
# =======================================================
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.ollama import OllamaEmbedding
from pathlib2 import Path
from custom_components import CustomVllmLLM, CustomSynthesizer
from custom_query_engine import OnceQueryEngine, ChatQueryEngine
from data_and_index import VectorIndex
from query_agents import create_tool_agent, create_top_agent
import nltk
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")

nltk.data.path.append('/mnt/e/PyCharm/PreTrainModel/nltk_data')
class LegalRAG:
    def __init__(self, llm_path):
        # 自定义大模型
        self.custom_model = CustomVllmLLM(llm_path)
        # 自定义响应器
        self.custom_synthesizer = CustomSynthesizer(self.custom_model)
        # 从文件构建索引，并保存
        self.index = VectorIndex()
        self.callback_manager = CallbackManager()

    def top_agent(self, file_paths: str):
        agents = {}
        for file_path in Path(file_paths).rglob('*'):
            file_name = file_path.stem
            file_path = str(file_path)

            # 自定义查询引擎, 构建单次对话向量查询引擎
            vector_index = self.index.create_vector_index(file_path)
            vector_engine = OnceQueryEngine(
                    vector_index.as_retriever(similarity_top_k=5),
                    self.custom_model
            )

            # 构建语义检索引擎, 这里使用直接构建的方式, 使用自定义的大模型
            summary_index = self.index.create_summary_index(
                    file_path,
                    self.custom_model
            )
            summary_engine = summary_index.as_query_engine(
                    llm=self.custom_model,
                    response_mode="tree_summarize"

            )

            # 构建工具agent
            tool_agent = create_tool_agent(
                    query_engine=vector_engine,
                    summary_engine=summary_engine,
                    name=file_name,
                    agent_llm=self.custom_model,
            )

            agents[file_name] = tool_agent

        top_agent_ = create_top_agent(
                agents,
                self.callback_manager,
                agent_llm=self.custom_model
        )

        return top_agent_


if __name__ == '__main__':
    # rag = LegalRAG('/media/xk/D6B8A862B8A8433B/data/qwen2_05b')
    # top_agent = rag.top_agent('/home/xk/PycharmProjects/llamaindexProjects/falv_rag/data')

    # rag = LegalRAG('/mnt/e/PyCharm/PreTrainModel/qwen2_05b')
    rag = LegalRAG('/mnt/e/PyCharm/PreTrainModel/qwen2_7b_instruct_awq_int4')
    top_agent = rag.top_agent('../data')

    # rag = LegalRAG(r'E:\PyCharm\PreTrainModel\qwen_7b_chat_int4')
    # top_agent = rag.top_agent('../data')
    res=top_agent.streaming_chat_repl()
    # print(res)
    # print('333')
