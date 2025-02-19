# -*- coding: utf-8 -*-
# @Time    : 2025/2/16 11:01
# @Author  : yblir
# @File    : query_agents.py
# explain  : 
# =======================================================
from llama_parse import LlamaParse
# from llama_index.readers.web import AsyncWebPageReader, BeautifulSoupWebReader, NewsArticleReader, SimpleWebPageReader
from llama_index.core.schema import Document, TextNode, MetadataMode
from typing import Any, Callable, Dict, List, Optional, Tuple
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer, \
    PromptTemplate, SummaryIndex, DocumentSummaryIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser, CodeSplitter, SentenceWindowNodeParser, \
    SimpleNodeParser, MarkdownElementNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from multiprocessing import freeze_support, set_start_method
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.agent.openai import OpenAIAgent
from queue import Queue
from pydantic import BaseModel

import pprint, chromadb, sys, os

from custom_query_engine import OnceQueryEngine, CustomVllmLLM, CustomSynthesizer




# 为单个文件创建agent
def create_tool_agent(file: str, name):
    # 数据加载器读取文档
    document = SimpleDirectoryReader(input_files=[file]).load_data()
    # 创建句子分割器, 对文档进行分割
    spliter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
    # 从句子分割器获得节点数据
    nodes = spliter.get_nodes_from_documents(document)

    # 将切分好的数据保存在向量库中,使用时直接从库中取
    if not os.path.exists(f"./storage/{name}"):
        print('Creating vector index...\n')
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # 向量存储索引, 只支持一种检索模式，就是根据向量的语义相似度来进行检索,
        # 对应的检索器类型为VectorIndexRetriever
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vector_index.storage_context.persist(persist_dir=f"./storage/{name}")
    else:
        print('Loading vector index...\n')
        storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{name}", vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    # todo 改为改写的自定义查询引擎
    query_engine = vector_index.as_query_engine(similarity_top_k=5)

    # Create a summary index
    # 文档摘要索引与向量存储索引的最大区别是，其不提供直接对基础Node
    # 进行语义检索的能力，而是提供在文档摘要层进行检索的能力，然后映射到基
    # 础Node。
    # summary_index = SummaryIndex(nodes)
    summary_index = DocumentSummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

    # 转换为工具, 供agent调用
    query_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=f'query_tool',
            description=f'Use if you want to query details about {name}'
    )
    summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_engine,
            name=f'summary_tool',
            description=f'Use ONLY IF you want to get a holistic summary of the documents.'
                        f' DO NOT USE if you want to query some details about {name}.'
    )

    # 创建一个tool agent
    tool_agent = ReActAgent.from_tools(
            tools=[query_tool, summary_tool],
            verbose=True,
            system_prompt=f"""You are a specialized agent designed to answer queries about {name}.
                          You must ALWAYS use at least one of the tools provided when answering a question;
                           do NOT rely on prior knowledge.DO NOT fabricate answer.""",
            # todo 用途?
            callback_manager=None
    )
    return tool_agent


# 为所有文件创建tool agent
def create_tool_agents(file_paths: Dict[str, str]):
    print('Creating document agents for all files...\n')
    tool_agents = {}
    for file_name, file_path in file_paths.items():
        tool_agents[file_name] = create_tool_agent(file_path, file_name)

    return tool_agents


def create_top_agent(tool_agents: Dict, callback_manager: CallbackManager):
    all_tools = []
    for file_name in tool_agents.keys():
        law_summary = (
            f" 这部分包含了有关{file_name}的信息. "
            f" 如果需要回答有关{file_name}的问题，请使用这个工具.\n"
        )

        tool = QueryEngineTool(
                # Agent 本身也是一种查询引擎，因此Agent 也可以作为一种查询
                # 引擎被包装成工具，供其他Agent 使用，从而实现了Agent 之间互相调用
                query_engine=tool_agents[file_name],
                metadata=ToolMetadata(
                        name=f"tool_{file_name}",
                        description=law_summary,
                ),
        )
        all_tools.append(tool)

    # 实现一个对象索引,用于检索 合适的tool
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
    # obj_index是否已向量持久化.
    if not os.path.exists(f"path/top/top"):
        storage_context = StorageContext.from_defaults()
        obj_index = ObjectIndex.from_objects(
                objects=all_tools,
                object_mapping=tool_mapping,
                index_cls=VectorStoreIndex,
                storage_context=storage_context
        )
        storage_context.persist(persist_dir=f"path/top/top")
    else:
        storage_context = StorageContext.from_defaults(
                persist_dir=f"path/top/top"
        )
        index = load_index_from_storage(storage_context)
        obj_index = ObjectIndex(index, tool_mapping)

    print('Creating top agent...\n')

    top_agent = ReActAgent.from_tools(
            tool_retriever=obj_index.as_retriever(similarity_top_k=3),
            verbose=True,
            system_prompt="""
                       你是一个被设计来回答关于法律问题的助手。
                       任何时候请选择提供的工具来回答问题。
                       不得依赖已有的知识。不要编造答案。""",
            callback_manager=callback_manager
    )
    return top_agent


if __name__ == '__main__':
    pass
