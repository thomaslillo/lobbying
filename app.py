import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from decouple import config
import pandas as pd
import json

## APP SETUP --------------------------------

# Access the API key
api_key = config('OPEN_API_KEY')
#openai.api_key = api_key

@st.cache_resource
def load_db():
    # set up the database collection
    chroma_client = chromadb.HttpClient(host='localhost', port='8000')
    # setup the embedding function to use OpenAI ada
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-ada-002")
    print(chroma_client.list_collections())    
    # get the collection
    return chroma_client.get_or_create_collection(name="ea_metadata", embedding_function=openai_ef)

@st.cache_data
def load_categories():
    df = pd.read_csv("DataPrep/Data/dataset_finished.csv", encoding='utf-8-sig')
    return df

# -- loading data

collection = load_db()
sidebar_df = load_categories()

## APP UI -----------------------------------

st.header('Welcome to EAVar Smart Search')
st.markdown("---")

st.markdown("*E.g. How many people are buying things at malls?*", 
            help="This is an example question"
            )

query = st.text_input("Provide a question and we'll provide variables!")
st.markdown("*We record these searches to help us build better products, don't enter personal information (we'll also auto-delete any we come across)!*")

# default filters is canada
filters = {'COUNTRY': 1, 'DATASETNAME': {'$in':[]}}
has_filters = False

with st.sidebar:
    st.text("")
    st.text("")
    st.markdown("#### Filter your search down:")
    st.text("")
    country = st.selectbox("Select your country ",("Canada","USA"))   
    vintage = st.multiselect("Select Year", options=["2021", "2022", "2023"]) 
    full_category_list = sidebar_df['Category'].unique().tolist()
    category = st.multiselect("Select a Category",full_category_list, help="If none are selected all categories will be searched")    

    st.button("Apply Filters")
        # has_filters = True
        # # Define filter conditions
        # if country == "Canada":
        #     country_code = 1
        # else:
        #     country_code = 0
        # condition1 = sidebar_df['COUNTRY'] = country_code  # Condition for 'Column1'
        # # Create a boolean mask to check if 'Column2' values are in 'values_to_match'
        # condition2 = sidebar_df[sidebar_df['Category'].isin(category)]

        # # Create an AND filter condition using the '&' operator
        # and_condition = condition1 & condition2

        # # Apply the filter to the DataFrame
        # filtered_df = sidebar_df[and_condition]

if query != "":
    if st.button("Submit"):

        if has_filters:
            results = collection.query(
                query_texts=[query],
                n_results=10,
                where=filters
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=10
            )
        
        # format the results
        metadata_df = pd.DataFrame((results['metadatas'])[0])
        documents_df = pd.DataFrame((results['documents'])[0])
        results_df = pd.concat([documents_df,metadata_df], axis=1)
        results_df = results_df.rename(columns={
            0: 'Variable Description',
            'DATASETNAME': 'Dataset (Will be Links)',
            'VARIABLENAME': 'Variable Code',
            'VINTAGE': 'Vintage',
            })
        results_df = results_df[['Variable Description','Dataset (Will be Links)','Variable Code','Vintage']]

        results_df['Vintage'] = results_df['Vintage'].astype(str)

        # display the results
        st.dataframe(results_df, hide_index=True)

        st.text("Contact us for access to these variables and more!")
        email = st.text_input("Email")
        st.button("Send")

        # format the results of the json
        #st.json(results)