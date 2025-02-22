# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 23:57
# @Author  : yblir
# @File    : data_and_index.py
# explain  : 
# =======================================================
import os
from loguru import logger
import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer, \
    PromptTemplate, SummaryIndex, DocumentSummaryIndex, load_index_from_storage, StorageContext, KeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser, CodeSplitter, SentenceWindowNodeParser, \
    SimpleNodeParser, MarkdownElementNodeParser

# Settings.llm=
# 在vector_store_index = VectorStoreIndex(node, storage_context=storage_context)时会隐式调用
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")
# 创建持久化的Chroma客户端
chroma = chromadb.PersistentClient(path="./chroma_db")
chroma.heartbeat()

collection = chroma.get_or_create_collection(name="legal_knowledge_rag")
# vector_store = ChromaVectorStore(chroma_collection=collection)
# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=collection)


# 创建存储上下文, 准备向量存储索引
# storage_context = StorageContext.from_defaults(vector_store=vector_store)


# 所有文件都从这里读取
class VectorIndex:
    def __init__(self):
        # if not os.path.exists(file_paths):
        #     raise ValueError('文件路径不存在')
        # self.nodes = self.read_data(file_paths)
        pass

    @staticmethod
    def read_data(file_path: str):
        # nodes = {}
        if not os.path.isfile(file_path):
            raise ValueError(f'{file_path} is not file')

        # 获得不带后缀的文件名
        # file_name = file_path.split(os.sep)[-1].split('.')[0]
        document = SimpleDirectoryReader(input_files=[file_path]).load_data()
        # 创建句子分割器, 对文档进行分割
        spliter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
        # 从句子分割器获得节点数据
        node = spliter.get_nodes_from_documents(document)

        # node_embedding = embed_model(node)
        # vector_store.add(node_embedding)
        # return {file_name: node}

        return node

    def create_vector_index(self, file_path: str):
        # 获得不带后缀的文件名
        file_name = file_path.split(os.sep)[-1].split('.')[0]
        node = self.read_data(file_path)

        # 将切分好的数据保存在向量库中,使用时直接从库中取
        if not os.path.exists(f"../chroma_db/vector_store_index/{file_name}"):
            logger.info(f'create vector index: {file_name}')
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # 向量存储索引, 只支持一种检索模式，就是根据向量的语义相似度来进行检索,
            # 对应的检索器类型为VectorIndexRetriever
            vector_store_index = VectorStoreIndex(node, storage_context=storage_context)
            vector_store_index.storage_context.persist(persist_dir=f"../chroma_db/vector_store_index/{file_name}")
        else:
            logger.info(f'load vector index: {file_name}')
            storage_context = StorageContext.from_defaults(
                    persist_dir=f"../chroma_db/vector_store_index/{file_name}",
                    vector_store=vector_store
            )
            vector_store_index = load_index_from_storage(storage_context=storage_context)

        return vector_store_index

    def create_keyword_index(self, file_path: str):
        # 获得不带后缀的文件名
        file_name = file_path.split(os.sep)[-1].split('.')[0]
        node = self.read_data(file_path)

        if not os.path.exists(f"../chroma_db/keyword_index/{file_name}"):
            logger.info(f'create keyword index: {file_name}')
            # 构造关键词表索引
            kw_index = KeywordTableIndex(node)
            kw_index.storage_context.persist(persist_dir=f"../chroma_db/keyword_index/{file_name}")
        else:
            logger.info(f'load keyword index: {file_name}')
            storage_context = StorageContext.from_defaults(persist_dir=f"../chroma_db/keyword_index/{file_name}")
            # 返回关键词检索器
            kw_index = load_index_from_storage(storage_context=storage_context)

        return kw_index

    def create_summary_index(self, file_path: str):
        # 获得不带后缀的文件名
        # file_name = file_path.split(os.sep)[-1].split('.')[0]
        node = self.read_data(file_path)

        # 文档摘要索引与向量存储索引的最大区别是，其不提供直接对基础Node
        # 进行语义检索的能力，而是提供在文档摘要层进行检索的能力，然后映射到基础Node。
        return DocumentSummaryIndex(node)

if __name__ == '__main__':
    # index=VectorIndex()
    pass
    # vector_index=index.create_vector_index('/home/xk/PycharmProjects/llamaindexProjects/《基于大模型的RAG应用开发与优化》源代码样例/src/12.3 端到端应用/multi-dos-agent/backend/data/上海市.txt')
    # print(vector_index)
    # keyword_index=index.create_keyword_index('/home/xk/PycharmProjects/llamaindexProjects/《基于大模型的RAG应用开发与优化》源代码样例/src/12.3 端到端应用/multi-dos-agent/backend/data/上海市.txt')

