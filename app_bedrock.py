import streamlit as st
import boto3
import json

# -------------------
# AWS Clients
# -------------------
region_name = 'us-east-1'
client = boto3.client('bedrock-agent-runtime', region_name=region_name)
llm_client = boto3.client("bedrock-runtime", region_name=region_name)

# -------------------
# Streamlit UI
# -------------------
st.title("Business Requirement Expander (RAG + LLM)")

# Model selection
model_id = st.selectbox(
    "Select a Model",
    [
        "meta.llama3-70b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "amazon.nova-micro-v1:0",
        "us.deepseek.r1-v1:0",
        "mistral.mistral-large-2402-v1:0"
    ],
    index=0
)

# User inputs
knowledge_base_id = st.text_input("Knowledge Base ID", value="2BHDRTHLT7")
query_text = st.text_area("Enter a short requirement:", value="Add validation on nominee age")

if st.button("Generate Expanded Requirement"):
    # -------------------
    # Retrieval
    # -------------------
    with st.spinner("Retrieving context from knowledge base..."):
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={'text': query_text},
            retrievalConfiguration={'vectorSearchConfiguration': {'numberOfResults': 10}}
        )

        retrieved_context = f"User query: {query_text}\n\nTop Retrieval Results:\n\n"
        i_context = ""

        for i, result in enumerate(response.get("retrievalResults", []), 1):
            text = result.get("content", {}).get("text", "").strip()
            source = result.get("location", {}).get("s3Location", {}).get("uri", "Unknown")
            score = result.get("score", 0.0)

            chunk = f"[{i}] Score: {score:.2f}\n{text}\nSource: {source}\n\n"
            retrieved_context += chunk
            i_context += text

        st.subheader("Retrieved Context")
        st.text_area("Context", retrieved_context, height=200)

    # -------------------
    # LLM Invocation
    # -------------------
    prompt = f"""You are a business analyst assistant. Based on the following context only, \
expand the given one-line requirement into a detailed business requirement or user story.
Don't ask follow up questions.

Context:
{i_context}

Requirement:
{query_text}

Output should include:
Description:
Roadmap:
Acceptance criteria:
Any roadblocks:


"""

    with st.spinner(f"Invoking model: {model_id}"):
        response = llm_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "temperature": 0.3
            })
        )
        output = json.loads(response['body'].read())
        expanded_req = output.get('generation', 'No response from model.')

        st.subheader("Expanded Requirement")
        st.write(expanded_req)
