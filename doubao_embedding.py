from typing import List
from langchain_core.embeddings import Embeddings
import dotenv
import os
from openai import OpenAI
# 加载环境变量文件
dotenv.load_dotenv()
# 设置环境变量
os.environ['OPENAI_API_KEY'] = os.getenv("ARK_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("ARK_BASE_URL")

class DoubaoEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """

        embeddings = self.client.embeddings.create(
            model="doubao-embedding-text-240715",
            encoding_format="float",
            input=texts
        )
           
        return [embeddings.embedding for embeddings in embeddings.data]
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]