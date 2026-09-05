from  dotenv  import  load_dotenv
load_dotenv()

from  langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from time import sleep
### CHRoma, FAISS, InMemory, Memory, VectorStore (Type of Vector Store )
from langchain_community.vectorstores import InMemoryVectorStore

import  streamlit as st;
import os

# api_key = st.secrets["GOOGLE_GEMINI_API_KEY"]
api_key = st.secrets.get("GOOGLE_GEMINI_API_KEY") or os.getenv(
    "GOOGLE_API_KEY"
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",google_api_key=api_key)

if "vector_db" not in st.session_state:
  st.session_state.vector_db = None

if "messages"  not  in st.session_state:
  st.session_state.messages = []

def document_process(path):

  ##  document loading process

  loader = PyPDFLoader(path)
  docs = loader.load()

  ## splitting process
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  docs = splitter.split_documents(docs)

  ## embedding and vector store process
  embeddings =  GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",google_api_key=api_key)
  vector_db = InMemoryVectorStore.from_documents(docs, embeddings)

  st.session_state.vector_db = vector_db

  st.session_state.document_uploaded = True

## chat process
# query = "What is the title of this course?"
# documents = vector_db.similarity_search(query, k  = 2)

# context = ""
# for  doc in documents:
#     context = context + doc.page_content + "\n\n"

# prompt = f"""
#     You are a helpful assistant that can answer questions about the given context.
#     Here is the context:
#     {context}
#     Here is the question:
#     {query}
#     """



# response = llm.invoke(prompt)

# if isinstance(response.content, list):
#     answer = "\n".join(
#         block["text"]
#         for block in response.content
#         if block.get("type") == "text"
#     )
# else:
#     answer = response.content

# print(answer)


## DDocument uploading UI
st.subheader("Document Q&A Chatbot - Ask Anything")

if "document_uploaded" not in st.session_state:
  st.session_state.document_uploaded = False

if not st.session_state.document_uploaded:
  file = st.file_uploader(label="Select your PDF File", type="pdf")
  if file:
    with open("uploaded_document.pdf", "wb") as f:
      f.write(file.getvalue())
    
    with st.spinner("Processing..."):
      document_process("./uploaded_document.pdf")
      

    st.markdown("Document  processed  successfully...")
    sleep(2)
    st.rerun()

## Chat UI

if  st.session_state.document_uploaded and st.session_state.vector_db:
  query =  st.chat_input("Ask Anything...")
  if  query:

    for message in st.session_state.messages:
      role = message["role"]
      content = message["content"]
      st.chat_message(role).markdown(content)

    st.session_state.messages.append({"role": "user", "content" : query})
    st.chat_message("user").markdown(query)
    documents =  st.session_state.vector_db.similarity_search(query, k  = 5)

    context = ""
    for doc in documents:
      context = context + doc.page_content + "\n\n"

    prompt = f"""You are helpful assistant and  you provide answers for the user's questions based on thhe  provided context.  context : {context} and questions : {query}"""

    result = llm.invoke(prompt)

    print(result.content)
    if isinstance(result.content, list):
      answer = "\n".join(
          block["text"]
          for block in result.content
          if block.get("type") == "text"
      )
    else:
      answer = result.content
    st.chat_message("ai").markdown(answer)
    st.session_state.messages.append({"role": "ai", "content" : answer})















