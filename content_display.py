import streamlit as st
import pandas as pd
import pinecone
import yaml
import openai
# from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

embed_model = "text-embedding-ada-002"

limit = 3750

# define a function to query the vector database
def semantic_search(query, k):
    # Create an embedding for our question
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    
    xq = res['data'][0]['embedding']

    # get relevant matches
    results = index.query(
        xq, 
        top_k=k, 
        include_metadata=True,
        filter={
            "version" : {"$eq": "Infoworks 5.4.2"}
        },
    )

    return results

# define a function to build a prompt based on semantic search results

def retrieve(query, results):

    contexts = [
        x['metadata']['text'] for x in results['matches']
    ]

    # build our prompt with the retrieved contexts included
    # If you do not know the answer, say 'I did not find anything in my documentation on that topic.'
    prompt_start = (
        "Answer the question based on the context below. If you do not know the answer, say 'I did not find enough info in documentation to give you an answer.'\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

# define a function to return answers

def complete(prompt):
  res = openai.Completion.create (
      engine = 'text-davinci-003',
      prompt = prompt,
      temperature = 0,
      max_tokens = 400,
      top_p = 1,
      frequency_penalty = 0,
      presence_penalty = 0,
      stop = None
  )
  return res['choices'][0]['text'].strip()

# load embedding model
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load config file
with open('./config/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

index_name = "iwx-semantic-search"

openai.api_key = config_data['openai']['openai_API_KEY']

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=config_data['pinecone']['pinecone_API_KEY'],
    environment=config_data['pinecone']['pinecone_ENV']
)

# connect to index
index = pinecone.Index(index_name)

# stats = index.describe_index_stats()
# s = index.describe_index_stats()
# st.write(str(s))

# image = Image.open('sunrise.jpg')

st.image('https://www.infoworks.io/wp-content/uploads/2022/09/logo-orig.svg')
        
st.write("#### Infoworks Assistant")

st.write("Hello, I am a helpful AI assistant that can answer your questions about Infoworks based on the official Infoworks documentation.")

search_criteria = st.text_input("How can I help you today?", label_visibility="visible")

if search_criteria:
    with st.spinner():
        results = semantic_search(search_criteria, 3)
        prompt = retrieve(search_criteria, results)
        # st.write("Found the docs.")

        answer_event = openai.Completion.create (
            engine = 'text-davinci-003',
            prompt = prompt,
            temperature = 0,
            max_tokens = 400,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0,
            stop = None,
            stream = True
        )
        # answer = complete(prompt)
        # answer2 = complete(search_criteria)
        
        # st.table(xr)
        # st.markdown(f"##### Answer without context")
        # st.markdown(f"{answer2}")
        # st.write("Finished formulating my answer.")
        st.markdown(f"##### Answer")
        answer_display = st.empty()
        references_display = st.empty()
        answer = ""

        for event in answer_event:
            answer += event['choices'][0]['text']
            answer_display.markdown(answer)


        with references_display.container():

            # st.markdown(f"{answer}")
            st.markdown(f"##### Related documentation.")
            docs = []
            for result in results['matches']:
                doc = {
                    "doc_url" : result['metadata']['doc_url'],
                    "version" : result['metadata']['version'],
                    "score" : result['score']
                }

                docs.append(doc)
                # st.write(f"{result['metadata']['doc_url']} | v: {result['metadata']['version']}")
                # st.write(type(result['metadata']))
            # st.write(results['matches'])
            df = pd.DataFrame.from_records(docs)
            st.dataframe(
                df,
                column_config={
                    "doc_url": st.column_config.LinkColumn(
                        "Infoworks Documentation",
                        help="Infoworks documentation",
                        validate="^https://[a-z]+\.docs\.infoworks\.io$",
                        max_chars=100,
                    )
                },
                hide_index=True,
            )

        # for x in xr.matches:
        #     st.divider()
        #    st.markdown(f"##### {x.metadata['text']}")


# What are the steps to sync metadata from Snowflake?
# List the steps to use the query as a table functionality
# List the prerequisites to onboard data from Teradata
# I want to ingest data from Teradata. What steps will I need to perform before getting started?
# I want to ingest data from Teradata. What steps will I need to perform before getting started? Please display the steps in a numbered list.


