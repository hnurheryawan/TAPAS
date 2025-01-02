import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from st_aggrid import AgGrid
import pandas as pd 
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# im = Image.open("ai-favicon.png")
# st.set_page_config(page_title="Table Summarization",
#     page_icon=im,layout='wide') 


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

st.markdown('<p style="font-family:sans-serif;font-size: 1.9rem;"> HertogAI Question Answering using TAPAS and ChatBot</p>', unsafe_allow_html=True)
st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'>Pre-trained TAPAS model runs on max 64 rows and 32 columns data. Make sure the file data doesn't exceed these dimensions.</p>", unsafe_allow_html=True)

tqa = pipeline(task="table-question-answering", 
                    model="google/tapas-large-finetuned-wtq",
                    temperature=0.5)

# Tambahkan model kecil untuk natural language generation
text_generator = pipeline('text-generation',
                        model='facebook/opt-125m',
                        max_new_tokens=50,
                        temperature=0.3)

# st.sidebar.image("ai-logo.png",width=200)
# with open('data.csv', 'rb') as f:
#         st.sidebar.download_button('Download sample data', f, file_name='Sample Data.csv')
file_name = st.sidebar.file_uploader("Upload file:", type=['csv','xlsx'])

if file_name is None:
    st.markdown('<p class="font">Please upload an excel or csv file </p>', unsafe_allow_html=True)
else:
    try:
        # Baca file dengan separator semicolon
        df = pd.read_csv(file_name, sep=';')
        
        # Konversi kolom numerik
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
        # Debug info untuk melihat data
        st.write("Original Data:")
        st.write(df)
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        
    # Initialize chat history in session state if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Copy data numerik sebelum konversi ke string
    df_numeric = df.copy()
    
    # Konversi ke string untuk TAPAS
    df = df.astype(str)
    
    grid_response = AgGrid(
        df.head(5),
        columns_auto_size_mode='FIT_CONTENTS',
        editable=True, 
        height=300, 
        width='100%',
    )

    # Add radio button for question type
    question_type = st.radio(
        "Select the type of question you want to ask:",
        ["Table Question", "General Question"],
        help="Choose 'Table Question' to ask about the data in the table above, or 'General Question' for any other questions."
    )

    # Display chat history
    if question_type == "ChatBot":
        # Chat container for messages
        chat_container = st.container()
        
        # Display all messages in chat container
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write(f'🧑 You: {message["content"]}')
                else:
                    st.write(f'🤖 Assistant: {message["content"]}')
        
        # Chat input at the bottom
        with st.container():
            # Create two columns for input and button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input("Your message:", key="user_input", label_visibility="collapsed")
            
            with col2:
                if st.button("Send", use_container_width=True):
                    if user_input:
                        # Add user message to chat
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        
                        # Generate response
                        try:
                            prompt = f"""Below is a conversation between a helpful assistant and a user.
                            The assistant gives helpful, detailed, and friendly responses.

                            User: {user_input}
                            Assistant: Let me help you with that. """
                            
                            response = text_generator(
                                prompt, 
                                max_new_tokens=100,  # Increased token length
                                temperature=0.7,     # Slightly increased temperature
                                do_sample=True,
                                top_p=0.95,
                                top_k=50,
                                num_return_sequences=1,
                                truncation=True,
                                repetition_penalty=1.2
                            )[0]['generated_text']
                            
                            # Better response cleaning
                            cleaned_response = response.split("Assistant: Let me help you with that.")[-1].strip()
                            if not cleaned_response or len(cleaned_response) < 10:
                                cleaned_response = response.split("User:")[-1].split("Assistant:")[-1].strip()
                            
                            # Fallback for empty or non-helpful responses
                            if not cleaned_response or len(cleaned_response) < 10:
                                cleaned_response = "I apologize, but I need to better understand your question. Could you please provide more details?"
                            
                            # Add assistant response to chat
                            st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}. Please try a different question.")
                            print(f"Error details: {str(e)}")
                        
                        # Remove these lines (146-150)
                        # Rerun to update the chat
                        # st.experimental_rerun()
            
            # Add a clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    else:
        question = st.text_input('Type your question')
        
        with st.spinner():
            if(st.button('Answer')):
                if question_type == "Table Question":
                    try:
                        raw_answer = tqa(table=df, query=question, truncation=True)
                        
                        # Tampilkan format TAPAS original
                        st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'> Raw Result: </p>",
                                   unsafe_allow_html=True)
                        st.success(raw_answer)
                        
                        # Generate natural response
                        processed_answer = raw_answer['answer'].replace(';', ' ')
                        row_idx = raw_answer['coordinates'][0][0]
                        col_idx = raw_answer['coordinates'][0][1]
                        column_name = df.columns[col_idx]
                        row_data = df.iloc[row_idx].to_dict()
                        
                        # Buat kalimat dari semua kolom secara dinamis
                        column_info = []
                        for col, value in row_data.items():
                            column_info.append(f"the {col} is {value}")
                        all_columns_text = ". ".join(column_info)
                        
                        prompt = f"""Question: '{question}'
                        Answer found: {processed_answer}
                        Complete information: {all_columns_text}
                        """
                        
                        natural_response = text_generator(prompt, max_length=100, temperature=0.3)[0]['generated_text']
                        
                        st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'> Natural Language Result: </p>",
                                   unsafe_allow_html=True)
                        st.success(f"""Analysis Results:
                        • Answer: {processed_answer}
                        
                        Suggested Questions:
                        {natural_response}
                        
                        Data Location:
                        • Row: {row_idx + 1}
                        • Column: {column_name}
                        
                        Additional Context:
                        • Full Row Data: {row_data}
                        • Query Asked: "{question}"
                        """)
                    except Exception as e:
                        st.warning("Please retype your question and make sure to use the column name and cell value correctly.")
                
                else:  # General Question
                    prompt = f"Answer this question: {question}"
                    
                    try:
                        natural_response = text_generator(prompt, max_length=100, temperature=0.3)[0]['generated_text']
                        cleaned_response = natural_response.replace(prompt, '').strip()
                        
                        st.markdown("<p style='font-family:sans-serif;font-size: 0.9rem;'> Chatbot Response: </p>",
                                   unsafe_allow_html=True)
                        st.success(cleaned_response)
                    except Exception as e:
                        st.error("Sorry, I couldn't generate a response. Please try asking another question.")
