import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatTongyi


"""LLM初始化"""


def build_llm():
    LLM_1 = ChatTongyi(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://ws-k66usu35y908m0bq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    return LLM_1


"""embedding model 初始化"""


def build_embeddings():
    llm_embeddings = DashScopeEmbeddings(
        model="qwen3.7-text-embedding",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    return llm_embeddings


def load_and_split():
    loader = TextLoader(r"D:/LLM_TEST/Own_project_NO1_v1/随手记.md", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def get_vector_store(split_docs, embeddings):
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    if vectorstore._collection.count() == 0:
        print("首次创建 Chroma 向量库...")

        BATCH_SIZE = 20
        for i in range(0, len(split_docs), BATCH_SIZE):
            batch_docs = split_docs[i:i + BATCH_SIZE]
            print(
                f"正在处理第 {i + 1} ~ {i + len(batch_docs)} 个 chunk"
            )
            vectorstore.add_documents(batch_docs)
    else:
        print("加载已有 Chroma 向量库...")

    print("向量库中的文档数量：", vectorstore._collection.count())
    return vectorstore


def rag_answer(question, vector_store, llm):
    results = vector_store.similarity_search(
        question,
        k=3,
    )

    texts = []
    for doc in results:
        texts.append(doc.page_content)

    context = "\n\n".join(texts)

    if not results:
        return "报意思呀~~未能检索到相关内容喏~~~笔记里暂时没有相关内容捏~~~"
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("RAG检索出来的结果：", context)

        prompt = f"""
        请根据下面的内容或者知识,总结性地、详细地回答用户的问题。
        只能依据"知识内容"回答；如果知识内容不足以回答这个问题，就直说"笔记里没有相关内容"，不许编造。
    
        知识内容：
        {context}
    
        问题：
        {question}
        """

        results_llm = llm.invoke(prompt)
        return results_llm.content


def main():
    llm = build_llm()
    embeddings = build_embeddings()
    split_docs = load_and_split()
    vector_store = get_vector_store(split_docs, embeddings)

    while True:

        answer_sent = input("请输入您的困扰喵~！我在听喵：")

        if answer_sent.lower() in ('quit', 'q', 'exit', 'Q', '退出'):
            print("好的喵！感谢使用")
            break
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(rag_answer(answer_sent, vector_store, llm))


if __name__ == "__main__":
    main()
