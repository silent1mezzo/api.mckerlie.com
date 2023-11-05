from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_markdown_docs():
    loader = DirectoryLoader(
        "api/docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )

    return loader.load()

def load_pdf_docs():
    loader = PyPDFLoader("api/docs/public/resume.pdf")

    return loader.load()

def ask_question(question):
    chain = load_qa_with_sources_chain(OpenAI(temperature=0.2))
    documents = load_markdown_docs()
    pdfs = load_pdf_docs()

    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(documents+pdfs)
    '''
    search_index = FAISS.from_documents(documents+pdfs, OpenAIEmbeddings())

    result = chain(
        {
            "input_documents": search_index.similarity_search(question, k=2),
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]
    return result

def split_answer(answer):
    return answer.strip().split("\n")[0]