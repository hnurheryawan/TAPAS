import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from st_aggrid import AgGrid
import pandas as pd 
from transformers import pipeline

st.set_page_config(layout="wide")

style = '''
    <style>
        body {background-color: #F5F5F5; color: #000000;}
        header {visibility: hidden;}
        div.block-container {padding-top:4rem;}
        section[data-testid="stSidebar"] div:first-child {
        padding-top: 0;
    }
     .font {                                          
    text-align:center;
    font-family:sans-serif;font-size: 1.25rem;}
    </style>
'''
st.markdown(style, unsafe_allow_html=True)

st.markdown('<p style="font-family:sans-serif;font-size: 1.9rem;"> HertogAI Question Answering using TAPAS</p>', unsafe_allow_html=True)
st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'>Pre-trained TAPAS model runs on max 64 rows and 32 columns data. Make sure the file data doesn't exceed these dimensions.</p>", unsafe_allow_html=True)

# Initialize TAPAS
tqa = pipeline(task="table-question-answering", 
              model="google/tapas-large-finetuned-wtq",
              device="cpu")

file_name = st.sidebar.file_uploader("Upload file:", type=['csv','xlsx'])

if file_name is None:
    st.markdown('<p class="font">Please upload an excel or csv file </p>', unsafe_allow_html=True)
else:
    try:
        df = pd.read_csv(file_name, sep=';')
        
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
        st.write("Original Data:")
        st.write(df)
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
    
    df_numeric = df.copy()
    df = df.astype(str)
    
    grid_response = AgGrid(
        df.head(5),
        columns_auto_size_mode='FIT_CONTENTS',
        editable=True, 
        height=300, 
        width='100%',
    )

    question = st.text_input('Type your question')
    
    with st.spinner():
        if(st.button('Answer')):
            try:
                raw_answer = tqa(table=df, query=question, truncation=True)
                
                st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'> Raw Result: </p>",
                           unsafe_allow_html=True)
                st.success(raw_answer)
                
                processed_answer = raw_answer['answer'].replace(';', ' ')
                row_idx = raw_answer['coordinates'][0][0]
                col_idx = raw_answer['coordinates'][0][1]
                column_name = df.columns[col_idx]
                row_data = df.iloc[row_idx].to_dict()
                
                st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'> Analysis Results: </p>",
                           unsafe_allow_html=True)
                st.success(f"""
                • Answer: {processed_answer}
                
                Data Location:
                • Row: {row_idx + 1}
                • Column: {column_name}
                
                Additional Context:
                • Full Row Data: {row_data}
                • Query Asked: "{question}"
                """)
            except Exception as e:
                st.warning("Please retype your question and make sure to use the column name and cell value correctly.")
