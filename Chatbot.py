import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from joblib import Parallel, delayed
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_openai import ChatOpenAI

from auth.login import login
from brain import get_embedding, format_docs, calculate_cosine_similarity_parallel
from connectdb import connect_to_postgresql, load_chroma, query_qas
from data_preprocessing import tien_xu_li
from typing import List, Optional
import timeit
from langchain_core.pydantic_v1 import BaseModel, Field

# from langchain.chains import RetrievalQA
os.environ[ 'TF_ENABLE_ONEDNN_OPTS' ] = '0'
load_dotenv()

def chat_component():
    # Streamlit app title
    st.title("Chatbot An toàn giao thông")

    # User input for the chat
    question = st.chat_input("Hỏi bất kỳ điều gì bạn muốn")
    return question

class Search(BaseModel):
    """Search over a database of job records."""

    queries: List[ str ] = Field(
        ...,
        description="Truy vấn riêng biệt để tìm kiếm",
    )
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
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),

        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    return document_chain
if __name__ == "__main__":
    connection = connect_to_postgresql()

    # st.dataframe(collection.get())

    is_logged_in = False

    if connection:
        is_logged_in = login()
        df = query_qas()
        # Close the database connection
        connection.close()
    else:
        st.warning("Failed to connect to PostgreSQL.")

    # Set up the OpenAI API key
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ[ "OPENAI_API_KEY" ] = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        temperature=0,
        max_tokens=4000
    )
    output_parser = PydanticToolsParser(tools=[ Search ])

    system = """Bạn có khả năng đưa ra các truy vấn tìm kiếm từ việc tách nhỏ câu hỏi nếu có nhiều ý thành những câu hỏi riêng lẻ
             giúp thông tin rõ ràng và quá trình tìm kiếm tốt hơn. 
            Nếu cần tra cứu hai thông tin riêng biệt, bạn được phép làm điều đó!
            Nếu không thể tách ra thì hãy trả về câu cũ"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

    vectorstore = load_chroma()
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    contextualize_q_system_prompt = """Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng \
            có thể tham chiếu ngữ cảnh trong lịch sử trò chuyện, tạo thành một câu hỏi độc lập \
            có thể hiểu được nếu không có lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \
            chỉ cần định dạng lại nó nếu cần và nếu không thì trả lại như cũ."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    SYSTEM_TEMPLATE = """
                   Trả lời câu hỏi người dùng dựa trên ngữ cảnh bên dưới và thêm nguồn tham khảo bên dưới nếu có trong lĩnh vực an toàn giao thông tại Việt Nam .
                    Nếu ngữ cảnh không chứa bất cứ thông tin liên quan đến câu hỏi, đừng làm gì hết chỉ cần trả lời 
                    "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn.":

                   Câu hỏi: {question}
                   Context: {context}
                   <citations>
                   {source}
                   </citations>
                   Trả lời:
                   """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input[ "question" ]


    @chain
    def custom_chain(question):
        response = query_analyzer.invoke(question)
        docs = [ ]
        for query in response.queries:
            new_docs = retriever.invoke(query)
            docs.extend(new_docs)
        return [ response, docs ]


    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | question_answering_prompt
            | llm
            | StrOutputParser()
    )

    prompt = st.session_state.get("prompt", [ {"role": "system", "content": "none"} ])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ ]
    question = chat_component()

    answer = ''
    # Display previous chat messages
    for message in prompt:
        if message[ "role" ] != "system":
            with st.chat_message(message[ "role" ]):
                st.write(message[ "content" ])
    # Handle the user's question
    if question:
        try:
            # Add the user's question to the prompt and display it
            prompt.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner():
                    handle_question = tien_xu_li(question)
                    chain = custom_chain.invoke(handle_question)
                    if len(chain[ 0 ].queries) > 1:
                        handle_query = " ".join(chain[ 0 ].queries)
                        print(chain[ 0 ].queries)
                    else:
                        handle_query = handle_question
                    print("handle_query", handle_query)
                    start_time = timeit.default_timer()
                    embed_query = get_embedding().embed_query(handle_question)
                    df[ 'cosine_similarity' ] = Parallel(n_jobs=-1)(
                        delayed(calculate_cosine_similarity_parallel)(x, embed_query) for x in df[ 'embedding_title' ])
                    elapsed_time = timeit.default_timer() - start_time
                    print("Embedding and df time:", elapsed_time)
                    # Tìm dòng có cosine_score lớn nhất
                    max_cosine_row = df.loc[ df[ 'cosine_similarity' ].idxmax() ]
                    result = ""
                    response = [ ]
                    source = [ ]
                    print(max_cosine_row)
                    if max_cosine_row.cosine_similarity > 0.85:
                        botmsg = st.empty()

                        doc = Document(
                            page_content=max_cosine_row.content, metadata={"source": max_cosine_row.source}
                        )
                        source = [ doc.metadata[ 'source' ] ]
                        print(doc)
                        start_time = timeit.default_timer()
                        ai_msg = retriever_df(llm).stream(
                            {"question": handle_query,
                            "source": source,
                            "chat_history": st.session_state.chat_history,
                            "context": [doc]}
                        )
                        elapsed_time = timeit.default_timer() - start_time
                        print("retrieval time:", elapsed_time)
                        
                    else:
                        botmsg = st.empty()
                        doc = chain[ 1 ]
                        print(doc)
                        start_time = timeit.default_timer()
                        ai_msg = rag_chain.stream(
                            {"question": handle_query,
                            "source": source,
                            "chat_history": st.session_state.chat_history,
                            "context": doc}
                        )
                        
                        elapsed_time = timeit.default_timer() - start_time
                        print("retrieval time:", elapsed_time)

                    for text in ai_msg:
                        response.append(text)
                        result = "".join(response).strip()
                        botmsg.write(result)
                st.session_state.chat_history.extend([ HumanMessage(content=question), AIMessage(content=result) ])

            print(st.session_state.chat_history)
            # Update the prompt with the assistant's response
            prompt.append({"role": "assistant", "content": result})

            # Store the updated prompt in the session state
            st.session_state[ "prompt" ] = prompt
        except Exception as e:
            st.error("An error occurred during processing.")
            st.error(str(e))