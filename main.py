import streamlit as st
import json
import boto3
from googleapiclient.discovery import build
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# Load environment variables - try loading from .env file first, then fall back to st.secrets
load_dotenv()

def get_env_variable(key):
    """Get environment variable from .env file or Streamlit secrets"""
    return os.getenv(key) or st.secrets.get(key)

# Function to fetch Google Search results
def google_search(query, api_key, cse_id, num_results=5):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return [{'snippet': item['snippet'], 'link': item['link']} for item in res['items']]
    except Exception as e:
        st.error(f"Error performing Google search: {str(e)}")
        return []

# Function to summarize text using AWS Bedrock
def summarize_text_with_aws(text, model_id):
    try:
        # Initialize the Bedrock Runtime client
        client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=get_env_variable('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=get_env_variable('AWS_SECRET_ACCESS_KEY')
        )

        # Prepare the request body for Claude
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": f"Please summarize the following search results and provide a comprehensive answer:\n\n{text}"
                }
            ],
            "temperature": 0.7,
            "top_p": 0.999,
        }

        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        # Parse the response
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']

    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDeniedException':
            st.error("AWS Access Denied. Please check your credentials.")
        elif e.response['Error']['Code'] == 'ValidationException':
            st.error("Invalid model ID or request format.")
        else:
            st.error(f"AWS Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Streamlit App UI
def main():
    st.set_page_config(page_title="Claudeplexity", page_icon=":mag:")
    st.title("Claudeplexity: AI-Powered Research Assistant")
    st.write("Get comprehensive answers with reliable sources")

    # Add a nice search box
    query = st.text_input(
        "Ask anything...",
        key="search_box",
        help="Enter your question and press Enter"
    )
    
    if query:
        # Load API credentials using the get_env_variable function
        api_key = get_env_variable('GOOGLE_API_KEY')
        cse_id = get_env_variable('GOOGLE_SEARCH_ENGINE_ID')
        model_id = get_env_variable('BEDROCK_MODEL_ID')

        if not all([api_key, cse_id]):
            st.error("Missing API credentials")
            return

        with st.spinner('Searching and analyzing...'):
            # Fetch Google Search results
            search_results = google_search(query, api_key, cse_id)
            
            if search_results:
                # Prepare text for summarization
                full_text = "\n\n".join([
                    f"Source ({result['link']}): {result['snippet']}"
                    for result in search_results
                ])
                
                # Get summary from AWS Bedrock
                summary = summarize_text_with_aws(full_text, model_id)
                
                if summary:
                    # Display results in a nice format
                    st.subheader("Answer")
                    st.write(summary)
                    
                    st.subheader("Sources")
                    for result in search_results:
                        with st.expander(f"Source: {result['link']}", expanded=False):
                            st.write(result['snippet'])
            else:
                st.warning("No search results found. Please try a different query.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")