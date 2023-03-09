import json
import altair as alt
import pandas as pd
from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
from snowflake.snowpark import version as v
from snowflake.snowpark.functions import call_udf
import streamlit as st
import snowflake.snowpark as sp
import json 

def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("creds.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

#Streamlit UI
st.title('Customer Life time Value Prediction')

st.text('')
csv_file = st.file_uploader(label='Upload CSV file with user data', type = 'csv')

def predict(inputs):

    
    Streamlit_data = session.table('Streamlit_data')
    
    snowdf_results = Streamlit_data.select(*inputs,
                    call_udf("clv_xgboost_udf",(*inputs)).alias('PREDICTION')
                    )

    return snowdf_results

# Display the contents of the uploaded file as a DataFrame
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
        st.write(df)
        
        session = create_session()
        session.use_warehouse('FE_AND_INFERENCE_WH')
        session.use_database('tpcds_xgboost')
        session.use_schema('demo')
        df = session.create_dataframe(df)
        df.write.mode('overwrite').save_as_table('Streamlit_data')
        prediction = predict(df).to_pandas()
        
        st.write(prediction)
        
        
        
    except Exception as e:
        st.write("Error reading CSV file:", e)
        