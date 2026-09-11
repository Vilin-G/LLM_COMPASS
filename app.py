import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
import hashlib
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class RAGState(TypedDict):
    question: str
    answer: str
    context: str
    grade: str
    router: str


def build_llm():
    LLM_1 = ChatOpenAI(
        model=os.getenv("CHAT_MODEL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    return LLM_1


def build_embeddings():
    llm_embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
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


def build_graph(vector_store, llm):

    def router_decision_node(state):
        question = state["question"]
        prompt = f"""
        你作为一个知识库问答工具的分支路由决策器，根据用户提出的问题，判断用户的问题是需要使用
        知识库的内容来回答。如果是，则返回“notes”，如果用户只是在问候或闲聊则返回“chat”。
        无论是何种情况，你都必须而且只能返回一个单一字符串chat 或者 notes。 
        问题：
            {question}
        """
        router = llm.invoke(prompt).content.strip().lower()
        if "notes" in router:
            router = "notes"
        elif "chat" in router:
            router = "chat"
        else:
            router = "chat"
        return {"router": router}

    def direct_answer_node(state):
        question = state["question"]
        prompt = f"""
        你是用户的学习助手，友好地回应用户的问题。
        问题：
            {question}
        """
        answer = llm.invoke(prompt).content
        return {"answer": answer}

    def retrieve_node(state):
        question = state["question"]

        results = vector_store.similarity_search(
            question,
            k=3,
        )

        if not results:  # 空库情况
            return {"context": ""}

        texts_blocks = []
        for doc in results:
            sources = doc.metadata.get("source")
            texts_blocks.append(f"【来源】：{sources}\n{doc.page_content}")

        context = "\n\n".join(texts_blocks)
        return {"context": context}

    def grade_node(state):
        question = state["question"]
        context = state["context"]
        prompt = f"""
                你是检索质量评估员，下面是用户问题和检索到的知识内容，判断这些内容是否足以回答问题，
                只准回答 yes 或 no。
                知识内容：
                    {context}
                问题：
                    {question}
                """
        grade = llm.invoke(prompt).content.strip().lower()
        if "yes" in grade:
            grade = "yes"
        else:
            grade = "no"
        return {"grade": grade}

    def answer_node(state):
        question = state["question"]
        context = state["context"]

        if not context.strip():
            return {"answer": "没有找到相关文档捏，无法生成答案喵。"}
        else:
            prompt = f"""
                    请根据下面的内容或者知识,总结性地、详细地回答用户的问题。
                    如果知识内容足以回答，请依据内容回答，并在结束时单独列出"参考来源"，只写你实际引用的文件名。
                    如果知识内容不足以回答这个问题，请直接回答"笔记里没有相关内容"，且不要输出参考来源。
                    绝对不准捏造内容。
                    
                    知识内容：
                    {context}

                    问题：
                    {question}
                    """

            results_llm = llm.invoke(prompt)
            answer = results_llm.content
            return {"answer": answer}

    def no_answer_node(state):
        return {"answer": "抱歉呐，笔记里暂时没有相关内容喵~"}

    builder = StateGraph(RAGState)
    builder.add_node("router", router_decision_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("answer", answer_node)
    builder.add_node("no_answer", no_answer_node)

    def grade_decision(state):
        return state["grade"]

    def router_decision(state):
        return state["router"]

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        router_decision,
        {"notes": "retrieve",
         "chat": "direct_answer",
         }
    )
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        grade_decision,
        {"yes": "answer",
         "no": "no_answer",
         }
    )

    builder.add_edge("answer", END)
    builder.add_edge("no_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


def main():
    llm = build_llm()
    embeddings = build_embeddings()
    split_docs = load_and_split()
    vector_store = get_vector_store(split_docs, embeddings)
    graph = build_graph(vector_store, llm)

    while True:

        answer_sent = input("请输入您的困扰喵~！我在听喵：\n")

        if answer_sent.lower() in ('quit', 'q', 'exit', 'Q', '退出'):
            print("好的喵！感谢使用")
            break
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        result = graph.invoke({"question": answer_sent})
        print(result["answer"])


if __name__ == "__main__":
    main()
