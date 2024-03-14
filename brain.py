from io import BytesIO
from pathlib import Path
from typing import Tuple, List
import PyPDF2
import chromadb.utils.embedding_functions as embedding_functions
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
# from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import chromadb
from data_preprocessing import tien_xu_li
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from scipy.spatial.distance import cosine
# Hàm để tính cosine similarity
def calculate_cosine_similarity(embed_text, embed_query):
    return 1 - cosine(Convert(embed_text), embed_query)
def get_embedding():
    model_name = "BAAI/bge-m3"
    encode_kwargs = {'normalize_embeddings': True}

    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )
    
    return bge_embeddings
def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    try:
        pdf = PyPDF2.PdfReader(file)
        output = []
        print(len(pdf.pages))
        for page in pdf.pages:
            text = page.extract_text()
            text = tien_xu_li(text)
            output.append(text)

        # print(output)
        print("Done", filename)
        return output, filename
    except Exception as e:
        # Handle the exception here, you can print an error message or perform other actions.
        return [], filename

# function to parse text files
def parse_text(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    text = file.getvalue().decode('utf-8')
    text = tien_xu_li(text)
    if '\n\n\n' in text:
        sections = text.split('\n\n\n')
    else:
        sections = text.split('\n\n')
    print("Done", filename)     
    return sections, filename

def text_to_docs(text: List[str], filename :str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks


def docs_to_index(docs):
    print("embedding...")
    try:
        Chroma.from_documents(collection_name="rag",
                              documents=docs,
                              embedding=get_embedding(),
                              persist_directory="./chromadb")
    except Exception as e:
        # Handle the exception, print the error message, and traceback
        print(f"An error occurred: {str(e)}")

    return True


def get_index_for_files(files, file_names):
    documents = []
    for file, file_name in zip(files, file_names):
        if file_name.endswith('.pdf'):
            text, filename = parse_pdf(BytesIO(file), file_name)
        else:
            text, filename = parse_text(BytesIO(file), file_name)
        documents = text_to_docs(text, filename)
    # for doc in documents:
    #     print(doc)
    flag = docs_to_index(documents)
    return flag

def load_predefined_files(folder_path):
    predefined_files = []
    file_names = []
    for file_path in Path(folder_path).glob('*.*'):
        if file_path.suffix in ['.pdf', '.txt']:
            with open(file_path, 'rb') as file:
                predefined_files.append(BytesIO(file.read()))
                file_names.append(file_path.name)
    return predefined_files, file_names


def Convert(string):
    string = string.replace("[",'').replace(']','')
    li = list(map(float, string.split(",")))
    return li

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def calculate_cosine_similarity_parallel(x, embed_query):
    return calculate_cosine_similarity(x, embed_query)