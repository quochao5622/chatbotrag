import os

import streamlit as st
from dotenv import load_dotenv
from joblib import Parallel, delayed
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from auth.login import login
from brain import get_embedding, calculate_cosine_similarity, format_docs, calculate_cosine_similarity_parallel
from connectdb import connect_to_postgresql
from data_preprocessing import tien_xu_li
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from langchain.chains import RetrievalQA

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()
def chat_component():
    # Streamlit app title
    st.title("Chatbot An toàn giao thông")
    
    # User input for the chat
    question = st.chat_input("Hỏi bất kỳ điều gì bạn muốn")
    return question

def build_retrieval_qa(llm):
    vectorstore = Chroma(persist_directory="./chromadb",
                         embedding_function=get_embedding(),
                         collection_name="rag")
    retriever = vectorstore.as_retriever(search_type="mmr")

    # qa_system_prompt = """
    # Bạn là một Trợ lý hữu ích, người trả lời các câu hỏi
    # liên quan đến an toàn giao thông ở Việt Nam.
    # Quan trọng không trả lời các câu hỏi ngoài an toàn giao thông.
    # Giữ câu trả lời ngắn gọn và đi vào vấn đề.
    # Quan trọng nếu có tiền phạt thì chỉ lấy tiền phạt đầu tiên được tìm thấy không lấy ở dưới.
    # Nếu ngữ cảnh không phù hợp với câu hỏi chỉ cần trả lời "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn." và
    # đừng cố trả lời thêm gì.
    # Câu hỏi: {question}
    # Answer:
    # context: {context}"""
    qa_system_prompt = """Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. \
    Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. \
    Nếu bạn không biết câu trả lời hoặc ngữ cảnh không liên quan đến câu hỏi,
    chỉ nói rằng "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn.":

    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(qa_system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x[ "context" ])))
            | prompt
            | llm
            | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

def retriever_df(llm):
    SYSTEM_TEMPLATE = """
       Trả lời câu hỏi người dùng dựa trên ngữ cảnh bên dưới và thêm nguồn tham khảo bên dưới.
       Nếu ngữ cảnh không chứa bất cứ thông tin liên quan đến câu hỏi, đừng làm gì hết chỉ cần trả lời 
       "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn.":

       <context>
       {context}
       </context>
       <citations>
       {source}
       </citations>
       """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages")

        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    return document_chain
def app():
    connection = connect_to_postgresql()

    # st.dataframe(collection.get())

    is_logged_in = False

    if connection:
        is_logged_in = login()
        # Close the database connection
        connection.close()
    else:
        st.warning("Failed to connect to PostgreSQL.")

    filepath = 'data/tu-van-pha-luat-du-thao-luat-trat-tu-atgtdb.csv'
        
    
    # Set up the OpenAI API key
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Create an instance of ChatOpenAI with desired settings
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0,
        max_tokens=4000
    )

    prompt = st.session_state.get("prompt", [ {"role": "system", "content": "none"} ])
    question = chat_component()
    df = pd.read_csv(filepath)
    answer = ''
    # Display previous chat messages
    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
    # Handle the user's question
    if question:
        try:
            chat_history = [ ]
            # Add the user's question to the prompt and display it
            prompt.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner():
                    print(11)
                    handle_question = tien_xu_li(question)
                    embed_query = get_embedding().embed_query(handle_question)
                    df[ 'cosine_similarity' ] = Parallel(n_jobs=-1)(delayed(calculate_cosine_similarity_parallel)(x, embed_query) for x in df[ 'embedding_title' ])

                    # Tìm dòng có cosine_score lớn nhất
                    max_cosine_row = df.loc[ df[ 'cosine_similarity' ].idxmax() ]
                    result = ""
                    response = [ ]
                    print(max_cosine_row)
                    if max_cosine_row.cosine_similarity > 0.85:
                        botmsg = st.empty()

                        doc = Document(
                            page_content=max_cosine_row.content, metadata={"source": max_cosine_row.source}
                        )
                        stream = retriever_df(llm).stream(
                            {
                                "context": [ doc ],
                                "source": [ doc.metadata[ 'source' ] ],
                                "messages": [
                                    HumanMessage(content=question)
                                ],
                            }
                        )
                        print(max_cosine_row.content)
                        for text in stream:
                            response.append(text)
                            result = "".join(response).strip()
                            botmsg.write(result)
                    else:
                        botmsg = st.empty()
                        for text in build_retrieval_qa(llm).stream(handle_question):
                            if 'answer' in text:
                                response.append(text[ 'answer' ])
                                result = "".join(response).strip()
                                botmsg.write(result)
                            else:
                                print(text)

            # Update the prompt with the assistant's response
            prompt.append({"role": "assistant", "content": result})

            # Store the updated prompt in the session state
            st.session_state["prompt"] = prompt
        except Exception as e:
            st.error("An error occurred during processing.")
            st.error(str(e))


if __name__ == "__main__":
    app()
