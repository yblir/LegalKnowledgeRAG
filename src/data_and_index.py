# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 23:57
# @Author  : yblir
# @File    : data_and_index.py
# explain  : 
# =======================================================
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

collection = chroma.get_or_create_collection(name="agentic_rag")
# vector_store = ChromaVectorStore(chroma_collection=collection)
# 使用嵌入式运行向量数据库数据库
client = chromadb.PersistentClient(path="./chroma_db")
client.heartbeat()
# todo
# 所有文件都从这里读取