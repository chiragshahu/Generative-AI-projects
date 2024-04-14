import os
from PIL import Image
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
def get_gemini_response(ques):
    response = model.generate_content(ques)
    return response.text

model_vision = genai.GenerativeModel("gemini-pro-vision")
def get_gemini_final_response(input,image):

    if image=="":
        return get_gemini_response(input)

    if input!="":
        response = model_vision.generate_content([input,image])
    else:
        response = model_vision.generate_content(image)
    
    return response.text

#________________Useful functions________________________________________________________

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])
#______________________________________________________________________________________________________________


st.header("Query about image")
input = st.text_input("Input: ",key="input")
upload_file = st.file_uploader("choose an image....",type=["jpg","jpeg","png"])
image = ""

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image,caption="upload Image",use_column_width=True)

pdf_docs = ""
pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

submit = st.button("Submit")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#If submit is clicked.
if submit and (input or image!="" or pdf_docs!=""):

    
    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Done")

    response = get_gemini_final_response(input,image)
    st.session_state['chat_history'].append(("You", input))

    st.subheader("Response is:")
    
    st.write(response)

    if image!="":
        st.session_state['chat_history'].append(("Image", image))
    
    st.session_state['chat_history'].append(("Bot", response))
    

st.subheader("The Chat History is")
    
for role, text in st.session_state['chat_history']:

    if isinstance(text, Image.Image):
        st.write("Image")
        width, height = text.size
        new_size = (int(width * 0.3), int(height * 0.3))
        text = text.resize(new_size)
        st.image(text)
    else:
        st.write(f"{role}: {text}")

st.write("_______________________________________________________________________________________________________________________________________________") 



# ____Adding Chat History_________________________________________________________________________________________________________

