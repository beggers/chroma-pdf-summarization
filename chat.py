# I wrote this in ~2 hours, don't @ me about code quality.
import os

from langchain.chains import VectorDBQA
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


CLIENT_NAME = 'chroma-is-so-cool'
DOC_PATH = './data/'


def main():
    data = load_data()
    db = init_db(data)
    chat_forever(db)


def init_db(data):
    print('inserting data...', len(data))
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(data, embedding_function)
    print('data inserted')
    return db


def load_data():
    print('loading data...')
    titles = os.listdir(DOC_PATH)
    data = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for t in titles:
        print('\t loading ', t)
        loader = UnstructuredPDFLoader(DOC_PATH + t)
        for d in loader.load_and_split(text_splitter):
            data.append(d)
    print('data loaded')
    return data


def chat_forever(db):
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=db)

    while True:
        query = input('Enter query: ')
        if query == 'exit':
            break
        print('\t', qa.run(query))
        print('-' * 70)


if __name__ == '__main__':
    main()
