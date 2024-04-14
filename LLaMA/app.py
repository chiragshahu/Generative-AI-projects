import PyPDF2
import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


@cl.on_chat_start
async def on_chat_start():

    # elements = [
    # cl.Image(name="image", display="inline", path=".jpg")
    # ]
    # Inform the user that processing has started
    msg = cl.Message(content="Ask me your queries related to the Health & Dieseases")#,elements=elements)
    await msg.send()

    # Read the PDF file
    pdf = PyPDF2.PdfReader(open('Data_For_Medical_Bot.pdf', 'rb'))
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="llama2-uncensored")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama2-uncensored"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = "Ask the questions!"
    await msg.update()
    
    #store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
     # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
         # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    #return results
    await cl.Message(content=answer, elements=text_elements).send()