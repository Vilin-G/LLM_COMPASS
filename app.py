import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
import hashlib
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

"""LLM初始化"""


def build_llm():
    LLM_1 = ChatOpenAI(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://ws-k66usu35y908m0bq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    return LLM_1


"""embedding model 初始化"""


def build_embeddings():
    llm_embeddings = OpenAIEmbeddings(
        model="qwen3.7-text-embedding",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://ws-k66usu35y908m0bq.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
    )
    return llm_embeddings


def note_fingerprint():
    print("正在生成笔记指纹...")
    combine = ""
    target_dir = Path(__file__).parent / "notes"
    for node_path in target_dir.glob("*.md"):
        content = node_path.read_text(encoding="utf-8")
        combine += node_path.name + content

    return hashlib.sha256(combine.encode("utf-8")).hexdigest()


def load_and_split():
    target_dir = Path(__file__).parent / "notes"
    docs1 = []
    for file_path in sorted(target_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        docs = [Document(page_content=content,
                         metadata={"source": file_path.name})]
        docs1.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
    )
    split_docs = text_splitter.split_documents(docs1)
    return split_docs


def get_vector_store(split_docs, embeddings):
    persist_directory = "./chroma_db"
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    current_fingerprint = note_fingerprint()
    meta = vectorstore._collection.metadata

    if ((meta is None
            or meta.get("fingerprint") != current_fingerprint)
            or vectorstore._collection.count() == 0):
        print("首次创建对应 Chroma 向量库...")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        BATCH_SIZE = 20
        for i in range(0, len(split_docs), BATCH_SIZE):
            batch_docs = split_docs[i:i + BATCH_SIZE]
            print(
                f"正在处理第 {i + 1} ~ {i + len(batch_docs)} 个 chunk"
            )
            vectorstore.add_documents(batch_docs)
        vectorstore._collection.modify(metadata={"fingerprint": current_fingerprint})  # 更新指纹便利贴

    else:
        print("加载已有 Chroma 向量库...")

    print("向量库中的文档数量：", vectorstore._collection.count())

    return vectorstore


def rag_answer(question, vector_store, llm):
    results = vector_store.similarity_search(
        question,
        k=3,
    )

    texts_blocks = []
    for doc in results:
        sources = doc.metadata.get("source")
        texts_blocks.append(f"【来源】：{sources}\n{doc.page_content}")

    context = "\n\n".join(texts_blocks)

    if not results:
        return "报意思呀~~未能检索到相关内容喏~~~笔记里暂时没有相关内容捏~~~"
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("RAG检索出来的结果：", context)

        prompt = f"""
        请根据下面的内容或者知识,总结性地、详细地回答用户的问题。
        只能依据"知识内容"回答；如果知识内容不足以回答这个问题，就直说"笔记里没有相关内容"，不许编造。
        回答结束时单独列一行"参考来源"，只能写它实际引用的【来源】标签里的文件名，没用到的不许写。
    
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

        answer_sent = input("请输入您的困扰喵~！我在听喵：\n")

        if answer_sent.lower() in ('quit', 'q', 'exit', 'Q', '退出'):
            print("好的喵！感谢使用")
            break
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(rag_answer(answer_sent, vector_store, llm))


if __name__ == "__main__":
    main()
