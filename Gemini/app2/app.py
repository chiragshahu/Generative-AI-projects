from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import google.generativeai as genai

## Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows
st.title("SQL Query generator")
database_name = st.text_input("DATABASE : ",key="Insert the Name of Database (case sensitive)")
table_name = st.text_input("TABLE : ",key="Insert the Name of Table inside database (case sensitive)")
details = st.text_input("DETAILS :", key= """write some details about database to explain LLMs : ", ex-"The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION ,which represents name, class, section of students""")
query = st.text_input("Ask you query: ", key="input")

sql_submit = st.button("Generate SQL Query")

if(sql_submit):

   prompt=[
        
    f'''
    You are an expert in converting English questions to SQL query, you are being give the prompt and 
    you have to convert it into SQL query code, some details which help you to do so are given

    \nHere is an overview of database : {details}

    \nSome exapmples of queries are : 

    \nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM {table_name};

    \nExample 2 - Give the whole database table,
    the SQL command will be something like this SELECT * FROM {table_name}; 

    \ninstructuion : The sql code should not have ``` in beginning or end and sql word in output

    '''
    ]
   
   response = get_gemini_response(query,prompt)
   response = read_sql_query(response,database_name+".db")


   for row in response:
    st.write(row)

st.write("___________________________________________________________")
# -------------------- YouTube Video URL to Text generator -----------------------------------------
st.title("Youtube Lecture Notes Maker")
from youtube_transcript_api import YouTubeTranscriptApi

prompt = """You are Yotube lecture note maker. You will be taking the transcript text
and providing detail notes of the entire video and providing the note with topics. Please provide the notes of the text given here:  """

def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    
def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

# st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Generate Notes of video lecture"):
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("# Detailed Notes:")
        st.write(summary)