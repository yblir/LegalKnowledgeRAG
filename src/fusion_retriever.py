# -*- coding: utf-8 -*-
# @Time    : 2025/2/18 上午11:37
# @Author  : yblir
# @File    : fusion_retriever.py
# explain  : 高级检索器：融合检索
# =======================================================
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer, \
    PromptTemplate, KeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
# from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores import chroma
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer, \
    PromptTemplate, SummaryIndex
# from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, DecomposeQueryTransform, \
    StepDecomposeQueryTransform
from llama_index.llms.openai import OpenAI
import pprint
from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.agent import AgentRunner
from llama_index.agent.openai import OpenAIAgentWorker, OpenAIAgent
from llama_index.agent.openai import OpenAIAgentWorker
from tqdm.asyncio import tqdm
import os
import nest_asyncio, asyncio
from custom_components import CustomVllmLLM
from llama_index.core.retrievers import QueryFusionRetriever


# 分别定义两个构造检索器的函数，一个基于向量索引，另一个基于关键词表索引
# def create_vector_index_retriever(name: str):
#     # 解析 Document 为 Node
#     city_docs = SimpleDirectoryReader(input_files=[f"../../data/citys/{name}.txt"]).load_data()
#     splitter = SentenceSplitter(chunk_size=300, chunk_overlap=0)
#     nodes = splitter.get_nodes_from_documents(city_docs)
#     # 存储到向量库 Chroma 中
#     collection = chroma.get_or_create_collection(name=f"agent_{citys_dict[name]}",
#                                                  metadata={"hnsw:space": "cosine"})
#     vector_store = ChromaVectorStore(chroma_collection=collection)
#     # 首次运行时构造向量索引，完成后进行持久化存储，以后直接加载
#     if not os.path.exists(f"./storage/vectorindex/{citys_dict[name]}"):
#         print('Creating vector index...\n')
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)
#         vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
#         vector_index.storage_context.persist(persist_dir=f"./storage/vectorindex/{citys_dict[name]}")
#     else:
#         print('Loading vector index...\n')
#         storage_context = StorageContext.from_defaults(persist_dir=f"./storage/vectorindex/{citys_dict[name]}",
#                                                        vector_store=vector_store)
#         vector_index = load_index_from_storage(storage_context=storage_context)
#
#     vector_retriever = vector_index.as_retriever(similarity_top_k=3)
#
#     return vector_retriever
#
#
# def create_kw_index_retriever(name: str):
#     city_docs = SimpleDirectoryReader(input_files=[f"../../data/citys/{name}.txt"]).load_data()
#     splitter = SentenceSplitter(chunk_size=500, chunk_overlap=0)
#     nodes = splitter.get_nodes_from_documents(city_docs)
#
#     if not os.path.exists(f"./storage/keywordindex/{citys_dict[name]}"):
#         print('Creating keyeword index...\n')
#         # 构造关键词表索引
#         kw_index = KeywordTableIndex(nodes)
#         kw_index.storage_context.persist(persist_dir=f"./storage/keywordindex/{citys_dict[name]}")
#     else:
#         print('Loading keyeword index...\n')
#         storage_context = StorageContext.from_defaults(persist_dir=f"./storage/keywordindex/{citys_dict[name]}")
#         # 返回关键词检索器
#         kw_index = load_index_from_storage(storage_context=storage_context)
#
#     return kw_index.as_retriever(num_chunks_per_query=5)


class FusionRetriever(BaseRetriever):
    def __init__(
            self,
            retrievers: List[BaseRetriever],
            similarity_top_k: int = 3,
    ) -> None:
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    # 查询转换并非融合检索的必需步骤，你可以直接对输入问题进行基于多个
    # 类型索引的融合检索与生成
    def rewrite_query(self, query: str, num: int = 3):
        """ 将query转为num个查询问题"""
        """根据输入问题生成多个问题用于检索, 使得检索出更丰富的上下问"""
        prompt_rewrite_temp = """\
        您是一个聪明的查询生成器。请根据我的输入查询生成多个搜索查询问题。
        生成与以下输入查询相关的{num_queries}个查询问题 \n
        注意每个查询占一行 \n
        我的查询：{query}
        生成查询列表：
        """
        prompt_rewrite = PromptTemplate(prompt_rewrite_temp)

        response = llm_openai.predict(
                prompt_rewrite, num_queries=num, query=query
        )

        # 假设LLM将每个查询放在一行上
        queries = response.split("\n")
        return queries

    # 构造一个使用多检索器进行多次查询的辅助方法，并采用异步的方式并行检索
    async def run_queries(self, queries, retrievers):
        tasks = []
        for query in queries:
            for i, retriever in enumerate(retrievers):
                tasks.append(retriever.aretrieve(query))

        task_results = await tqdm.gather(*tasks)

        results_dict = {}
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result

        return results_dict

    # 3.使用 RRF 算法给检索出的多个 Node 重排序，并返回排序结果中的前 K 个 Node。下面是一个通用的算法
    def rerank_results(self, results_dict, similarity_top_k: int = 3):
        """Fuse results."""
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        text_to_node = {}

        # compute reciprocal rank scores
        for nodes_with_scores in results_dict.values():
            for rank, node_with_score in enumerate(
                    sorted(
                            nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
                    )
            ):
                text = node_with_score.node.get_content()
                text_to_node[text] = node_with_score
                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + k)

        # sort results
        reranked_results = dict(
                sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # adjust node scores
        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            reranked_nodes.append(text_to_node[text])
            reranked_nodes[-1].score = score

        return reranked_nodes[:similarity_top_k]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 查询转换，丰富查找内容
        queries = self.rewrite_query(query_bundle.query_str, num=3)
        # 使用异步方式进行检索
        results_dict = asyncio.run(
                self.run_queries(queries, self._retrievers)
        )
        # 检索结果排序
        final_results = self.rerank_results(
                results_dict,
                similarity_top_k=self._similarity_top_k
        )

        return final_results


# 使用自定义融合检索器
# def run_main():
#     query = "南京市有多少人口，是怎么分布的？"
#
#     #两个检索器
#     vector_retriever = create_vector_index_retriever('南京市')
#     kw_retriever = create_kw_index_retriever('南京市')
#
#     #创建融合检索器
#     fusion_retriever = FusionRetriever([vector_retriever, kw_retriever], similarity_top_k=3)
#
#     #查询引擎
#     query_engine = RetrieverQueryEngine(fusion_retriever)
#
#     #查询
#     response = query_engine.query(query)
#     pprint_response(response, show_source=True)


# 使用llamaindex官方封装的融合检索器
# def fusion_retriever():
class FusionRetriever2:
    def __init__(self, retrievers,retriever_llm, similarity_top_k=3, num_queries=4):
        # 两个检索器
        # vector_retriever = create_vector_index_retriever('南京市')
        # kw_retriever = create_kw_index_retriever('南京市')
        # QueryFusionRetriever封装全面，有子问题转换器， mode是只融合检索后方法排序方法
        # 可以为当前融合检索指定大模型，否则使用全局模型
        fusion_retriever = QueryFusionRetriever(
                retrievers=retrievers,  # [vector_retriever,kw_retriever]
                llm=retriever_llm,
                similarity_top_k=similarity_top_k,
                num_queries=num_queries,  # set this to 1 to disable query generation
                mode=FUSION_MODES.RECIPROCAL_RANK,  # "reciprocal_rerank",
                use_async=True,
                verbose=True
        )

        # 查询引擎,不再外面暴露
        self._query_engine = RetrieverQueryEngine(fusion_retriever)

    def query(self, query_question: str):
        return self._query_engine.query(query_question)
    # 查询
    # response = query_engine.query(query)
    # pprint_response(response)


if __name__ == '__main__':
    retriever_llm_ = CustomVllmLLM('/mnt/e/PyCharm/PreTrainModel/qwen2_7b_instruct_awq_int4')
    a = FusionRetriever2(retriever_llm_, 3, 4)
    a.query('dd')
