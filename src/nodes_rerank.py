# -*- coding: utf-8 -*-
# @Time    : 2025/2/19 上午11:03
# @Author  : yblir
# @File    : nodes_rerank.py
# explain  : 检索结果后处理，对检索结果进行排序
# =======================================================
import requests
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class BgeRerank(BaseNodePostprocessor):
    url: str = Field(description="Rerank server url.")
    top_n: int = Field(description="Top N nodes to return.")

    def __init__(self, top_n: int, url: str):
        super().__init__(url=url, top_n=top_n)

    # 调用 TEI 的 Rerank 模型服务
    def rerank(self, query, texts):
        url = f"{self.url}/rerank"
        request_body = {
            "query"   : query,
            "texts"   : texts,
            "truncate": False
        }
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to rerank, detail:{response}")
        return response.json()

    @classmethod
    def class_name(cls) -> str:
        return "BgeRerank"

    # 实现 Rank 节点后处理器的接口
    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        if len(nodes) == 0:
            return []
        # 调用 Rerank 模型
        texts = [node.text for node in nodes]
        results = self.rerank(
                query=query_bundle.query_str,
                texts=texts,
        )
        # 组装并返回 Node
        new_nodes = []
        for result in results[0: self.top_n]:
            new_node_with_score = NodeWithScore(
                    node=nodes[int(result["index"])].node,
                    score=result["score"],
            )
            new_nodes.append(new_node_with_score)

        return new_nodes


if __name__ == '__main__':
    # 单独使用
    # 构造自定义的节点后处理器
    customRerank = BgeRerank(url="http://localhost:8080", top_n=2)
    # 测试处理 Node
    rerank_nodes = customRerank.postprocess_nodes(nodes, query_str='百度文心一言的逻辑推理能力怎么样?')

    # 在构造查询引擎时直接使用
    query_engine = vector_index.as_query_engine(
            similarity_top_k=3,
            node_postprocessors=[customRerank],
    )
