# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import streamlit as st
import os
import re
from datetime import datetime
from pymilvus import(connections,Collection, MilvusClient)
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('MLV_URL')
port = os.getenv('MLV_PORT')
apikey = os.getenv('MLV_API_KEY')
apiuser = os.getenv('MLV_API_USER')
collection_name = os.getenv('MLV_COLLECTION')

api_endpoint = os.getenv('WSX_API_URL')
api_key = os.getenv('IAM_API_KEY')
project_id = os.getenv('WSX_PROJECT_ID')

wxai_credentials = {
    "url": api_endpoint,
    "apikey": api_key   
}

# Model Parameters
params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 4,
        GenParams.MAX_NEW_TOKENS: 500,
        # GenParams.TEMPERATURE: 0,
        GenParams.REPETITION_PENALTY: 1.05,
        # GenParams.STOP_SEQUENCES: ["\n\n"]
}

wxai_modelname = 'ibm/granite-13b-chat-v2'
def query_milvus(query, subject, num_results=5):
    print('QUERRY:', query, subject)
    # Vectorize query
    query_embeddings = embedding_model.encode(query, normalize_embeddings=True).tolist()
     
    # Search
    search_params = {
        "metric_type": "L2", 
        "params": {"nprobe": 20},
    }
    
    filter=f"subject == \'{subject}\'"    
    
    results = client.search(
        collection_name=collection_name,
        data=[query_embeddings], 
        # anns_field="embedding", 
        search_params=search_params,
        limit=num_results,
        output_fields=['content'],
        filter=filter
    )    
    return results

def remove_patterns(text):
    # Regular expression pattern to match text between <| and |>
    pattern = r'<\|.*?\|>'
    # Substitute the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

st.set_page_config(
    page_title='Watsonx Challenge 2024 - RAG to get FAQ',
    layout="wide",  # Set the layout to wide
    initial_sidebar_state="auto"  # Set the initial state of the sidebar
)

# Define the CSS style
css = """
{visibility: hidden;}
footer {visibility: hidden;}
body {overflow: hidden;}
data-testid="ScrollToBottomContainer"] {overflow: hidden;}
section[data-testid="stSidebar"] {
    width: 350px !important; # Set the width to your desired value
}
"""

# Display the dataframe with the custom CSS style
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# -------------------------------------------------
st.subheader('Watsonx Challenge 2024 - RAG to get FAQ', divider='rainbow')
st.sidebar.subheader('SWG Prague 2024 - wx.Challenge team', divider='grey')

# numberOfOptions = st.sidebar.radio("Počet variant odpovědí", options = [3,4,5,6,7,8])
topics = ["IBM watsonx Assistant", "IBM FlashSystem", "IBM Maximo", "Hamlet"]
subject = st.sidebar.selectbox(
    "Knowledge base selection",
    topics,
    help="Select KB",
)
language = "en"

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.subheader('', divider='grey')

message = """
Martin Ryšánek / D&AI (c) 2024  
"""
st.sidebar.markdown(message)

st.markdown("##### Write your query to RAG system for given Knowledge base")
question_text = st.text_input("Your Query", "")
   
pressed = st.button("Send Query to RAG")

if pressed:
    # print ('A', datetime.now())
    if "EMBEDDING" not in st.session_state:
        model_name = "intfloat/multilingual-e5-base"
        embedding_model = SentenceTransformer(model_name)
        st.session_state.EMBEDDING = embedding_model
    else:
        embedding_model = st.session_state.EMBEDDING

    # print ('B',datetime.now())        
    # print ('C', datetime.now())    
    if "MLV_CLIENT" not in st.session_state:
        client_config = 'https://' + apiuser + ':' + apikey + '@' + url + ':' + str(port)
        client = MilvusClient(client_config)
        st.session_state.MLV_CLIENT = client
    else:
        client = st.session_state.MLV_CLIENT
    # print ('D', datetime.now())    
    results = query_milvus(question_text, subject, num_results=6)
    # print ('E', datetime.now())    
    relevant_chunks = []
       
    for result in results[0]:    
        # print(f"id: {result['id']}")
        # print(f"distance: {result['distance']}")
        text = result['entity'].get('content')
        # print(f"id[{result['id']}]: {text}")        
        relevant_chunks.append(text)        
        
    prompt_input = """<|start_header_id|>system<|end_header_id|>

You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.

Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.

When returning code blocks, specify language.

You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe generative AI using emojis.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

**Generative AI** 🤖💻
=====================

### What is it? 🤔

Generative AI is a type of artificial intelligence that can create new, original content 🎨, such as:

* Images 📸
* Music 🎵
* Text 📝
* Videos 📹

### How does it work? 🔧

Generative AI uses complex algorithms and machine learning techniques to analyze existing data and patterns, and then generates new content based on that analysis 📊.

### Examples 📊

* **Artistic creations**: Generative AI can create original paintings, sculptures, and other forms of art 🎭.
* **Music generation**: Generative AI can compose music, create beats, and even generate entire songs 🎵.
* **Text generation**: Generative AI can generate text, such as articles, stories, and even entire books 📖.
* **Video generation**: Generative AI can create videos, such as animations, movies, and even entire TV shows 📺.

### Applications 📈

* **Content creation**: Generative AI can help create new content for various industries, such as entertainment, marketing, and education 📊.
* **Data augmentation**: Generative AI can be used to augment existing data, making it more diverse and representative 📊.
* **Research**: Generative AI can be used to simulate real-world scenarios, test hypotheses, and explore new ideas 🔬.

### Challenges 🚨

* **Bias and ethics**: Generative AI can perpetuate biases and ethical issues if not designed and trained properly 🚫.
* **Quality and accuracy**: Generative AI can struggle to match the quality and accuracy of human-created content 📊.
* **Control and understanding**: Generative AI can be difficult to control and understand, as it can create complex and unpredictable outputs 🤯.

### Future directions 🔜

* **Advancements in algorithms**: Generative AI algorithms will continue to improve, enabling more sophisticated and realistic content creation 📊.
* **Increased adoption**: Generative AI will be adopted across various industries, leading to new applications and use cases 📈.
* **Ethical considerations**: Generative AI will require careful consideration of ethical implications, ensuring responsible development and deployment 🚫."""


    context = "\n\n".join(relevant_chunks)
    question = f"""Consider strictly next text as context for given question.
Context of the question:
{context}

Answer following question and use above mentioned context.
Question:
{question_text}
"""
    formattedQuestion = f"""<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    prompt = f"""{prompt_input}{formattedQuestion}"""
    
    st.write("")
    # print ('F', datetime.now())   
    st.markdown("##### RAG query response using watsonx.ai model")
    loading_text = st.text("Quering RAG platform ...")      
    if "WXAI_MODEL" not in st.session_state:
        wxai_model = Model(
                model_id=wxai_modelname, 
                params=params, credentials=wxai_credentials, 
                project_id=project_id
        )   
        st.session_state.WXAI_MODEL = wxai_model
    else:
        wxai_model = st.session_state.WXAI_MODEL
    # print ('G', datetime.now())   
    response = wxai_model.generate_text(prompt=prompt, guardrails=False)
    # print ('H', datetime.now())   
    loading_text.empty()
    
#     st.markdown("""<style>
# p, div, span, li, body, strong {
#   font-family: Verdana, sans-serif;
#   font-size: 12px;
# }
# </style>
# """ + response, unsafe_allow_html=True)
    st.markdown(remove_patterns(response))
