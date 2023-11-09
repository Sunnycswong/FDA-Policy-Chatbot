## Import Library

import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)
from azure.storage.blob import BlobServiceClient
from langchain.chains import LLMChain
from langchain.llms import AzureOpenAI 
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import CosmosDBChatMessageHistory
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
import json

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
#from langchain.retrievers import AzureCognitiveSearchRetriever
from langdetect import detect
from langchain.prompts import PromptTemplate
import re
# Create chain to answer questions
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain

# Import Azure OpenAI
from langchain.llms import AzureOpenAI 
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

#import textwrap
import logging

# setting up credentials
os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "acs-fda-paid" # replace with yours search service name
os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = "VZhK9GGzAJ625kSvDXUpBo1CmiIfH6Ou64EDhoiSczAzSeADco5j" # replace with your api key
os.environ["AZURE_INDEX_NAME"] = "fda-index" #"namfung-finance-chatbot" # 
# end setting up credentials

#Free version of acs
#index_name = "your-index-name"
#search_service = "acs-testing-sunny"
#search_api_key = "oygYftyrBXiWoDLoZatDKNSLFttn9frM6DE4XlSb7kAzSeBR01eY"

#Paid version of acs
#index_name = "fda-index"
#search_service = "acs-fda-paid"
#search_api_key = "VZhK9GGzAJ625kSvDXUpBo1CmiIfH6Ou64EDhoiSczAzSeADco5j"


def initialize_vector_store():
    # set up index name 
    index_name = os.environ["AZURE_INDEX_NAME"] 
    
    # set up openai environment
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://pwcjay.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_KEY"] = "f282a661571f45a0bdfdcd295ac808e7"

    model: str = "text-embedding-ada-002"
    search_service = os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"]
    search_api_key = os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"]
    vector_store_address: str = f"https://{search_service}.search.windows.net"
    vector_store_password: str = search_api_key
    

    # define embedding model for calculating the embeddings
    model: str = "text-embedding-ada-002"
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
    embedding_function = embeddings.embed_query

    # define schema of the json file stored on the index
    fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_configuration="default",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Additional field to store the title
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="page",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="website_url",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
        ]    
    
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embedding_function,
        fields=fields,
    )
    
    return vector_store

def azure_search_by_index(question, index_name):

    # set up openai environment
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://pwcjay.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_KEY"] = "f282a661571f45a0bdfdcd295ac808e7"

    model: str = "text-embedding-ada-002"
    search_service = os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"]
    search_api_key = os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"]
    vector_store_address: str = f"https://{search_service}.search.windows.net"
    vector_store_password: str = search_api_key
    

    # define embedding model for calculating the embeddings
    model: str = "text-embedding-ada-002"
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
    embedding_function = embeddings.embed_query

    # define schema of the json file stored on the index
    fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_configuration="default",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Additional field to store the title
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="page",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="website_url",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
        ]    
    
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embedding_function,
        fields=fields,
    )

    relevant_documentation = vector_store.similarity_search(query=question, k=1, search_type="similarity")
    
    context = "\n".join([doc.page_content for doc in relevant_documentation])[:10000]

    lang = detect(context)

    #print(doc)
    #print(context)
    #print(relevant_documentation)
    source = relevant_documentation[0].metadata['source']
    #page_no = relevant_documentation[0].metadata['page']
    website_url = relevant_documentation[0].metadata['website_url']
    
    page_no = ""
    for doc in relevant_documentation:
        page_no = page_no + "," + doc.metadata['page'] 
    
    #print(relevant_documentation[0])
    #print(source)
    #print(page_no)
    #print(website_url)
    #return str(context), source, website_url, lang, page_no
    # just return 10 documents (i.e. pages) if number of pages return from the search result > 10
    if len(relevant_documentation) > 10:
        relevant_documentation = relevant_documentation[0:9]
    else:
        relevant_documentation = relevant_documentation
    return relevant_documentation, source, website_url, lang, page_no


# helper function to extract page number
def extract_page_no(string):
    if "[Page" in string:
        print(re.findall('\[Page:.*\]', string)[0].split('Page:')[1])
        return re.findall('\[Page:.*\]', string)[0].split(':')[1].split("]")[0].strip()
    else:
        return "/"

def extract_answer(string):
    if "[Page" in string:
        return string.split("[Page")[0]
    else:
        return string
def generate_prompt_with_history():
    prompt_template_string="""
    Follow exactly these 6 steps:
    1. Read the context below and aggregrate this data
    Context : {context}
    2. Answer the question using only this context and the chat history below
    3. Answer the question in less than 200 words
    4. Please provide the page number of the pages where your answer are based on at the end of your response
    5. Please provide the page numbers in the following output format: [Page: 1, 2, 3]
    6. Allow the chat continue by following Chat History
    
    Chat History: {chat_history}

    User Question: {question}


    If you don't have any context and are unsure of the answer, reply that you don't know about this topic.
    """
    prompt_template = PromptTemplate(template = prompt_template_string, input_variables=["context", "question", "chat_history"])

    return prompt_template

def generate_prompt_chi_with_history():
    prompt_template_string="""
    指令：
    1. 你必须只根据以下文本的内容及谈话记录回答提问者的询问。
    2. 如果不懂得回答或文本没有资料，请回答“对不起，我不懂得回答这个问题。”
    3. 请以少於200字回答问题。
    4. 请在你的回答后提供你用以回答的文本的页数。格式示例：[Page: 1]
    5. 通过谈话记录允许聊天继续
    
    文本：{context}

    谈话记录：{chat_history}

    #####

    问题：{question}

    """
    prompt_template = PromptTemplate(template = prompt_template_string, input_variables=["context", "question", "chat_history"])

    return prompt_template
    
# helper function to extract page number
def extract_page_no(string):
    if "[Page" in string:
        print(re.findall('\[Page:.*\]', string)[0].split('Page:')[1])
        return re.findall('\[Page:.*\]', string)[0].split(':')[1].split("]")[0].strip()
    elif "(Page:" in string: # handling for exception
        print(re.findall('\(Page:.*\)', string)[0].split('Page:')[1])
        return re.findall('\(Page:.*\)', string)[0].split(':')[1].split(")")[0].strip()
    else:
        return "/"

def extract_answer(string):
    if "[Page" in string:
        return string.split("[Page")[0]
    else:
        return string
        
        
def get_web_url(source):
    if source == "CFR-2022-title21-vol4-chapI-subchapC.pdf":
        website_url = "https://www.govinfo.gov/content/pkg/CFR-2022-title21-vol4/pdf/CFR-2022-title21-vol4-chapI-subchapC.pdf"
    elif source == "CFR-2022-title21-vol5-chapI-subchapD.pdf":
        website_url = "https://www.govinfo.gov/content/pkg/CFR-2022-title21-vol5/pdf/CFR-2022-title21-vol5-chapI-subchapD.pdf"
    elif source == "Bioavailability-and-Bioequivalence-Studies-Submitted-in-NDAs-or-INDs-—-General-Considerations.pdf":
        website_url = "https://www.fda.gov/media/88254/download"        
    else:
        website_url = "/"
    return website_url    


def llm_pipeline_with_history(question,sessionId):
    # set up index name 
    index_name = os.environ["AZURE_INDEX_NAME"] 

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pwcjay.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "f282a661571f45a0bdfdcd295ac808e7"


    # retrieve information from Azure Search
    relevant_docs, source, website_url, language, page_no = azure_search_by_index(question, index_name)

    language = detect(question)

    # Both Eng
    if language == "cn":
    #Chinese prompt
        # QA_CHAIN_PROMPT = generate_prompt_chi_with_history()
        QA_CHAIN_PROMPT = generate_prompt_with_history()
    else:
        QA_CHAIN_PROMPT = generate_prompt_with_history()

    # use AzureChatOpenAI 
    llm = AzureChatOpenAI(deployment_name="gpt-35-16k", temperature=0,
                        openai_api_version="2023-05-15", openai_api_base="https://pwcjay.openai.azure.com/")


    """    # set up chat history database credentials
    os.environ["COSMOS_ENDPOINT"] = "https://gpt-demo-chat-history.documents.azure.com:443/"
    os.environ["COSMOS_KEY"] = "AZEhMpW4YD3t7iEMgp9at48S7f5ZjvnahUqJMYjMjMpH2QH2wiYBL97RdX7AqL3CMQcGGhbdAFHvACDbDDwMyA=="
    ENDPOINT = os.environ["COSMOS_ENDPOINT"]
    KEY = os.environ["COSMOS_KEY"]
    DATABASE_NAME = "sino_demo"
    CONTAINER_NAME = "sino_chat_history"""
    
    # set up chat history database credentials
    os.environ["COSMOS_ENDPOINT"] = "https://acs-testing-sunny.documents.azure.com:443/"
    os.environ["COSMOS_KEY"] = "UyxAxYPy6nhqLoTVhbs7C8NknhHoaRJuFkBaZramSAEPzNsHU0dhanOTRr2AJOjtqA1m0d5N3ujkACDbNmxrAQ=="
    ENDPOINT = os.environ["COSMOS_ENDPOINT"]
    KEY = os.environ["COSMOS_KEY"]
    DATABASE_NAME = "acs-fda-sunny"
    CONTAINER_NAME = "acs-fda-cosmo"

    # session id, to be provided by frontend
    sessionId = sessionId
    user_id = "guest"

    history = CosmosDBChatMessageHistory(
        cosmos_endpoint = ENDPOINT,
        cosmos_database = DATABASE_NAME,
        cosmos_container = CONTAINER_NAME,
        credential = KEY,
        session_id = sessionId,
        user_id = user_id,
    )

    history.prepare_cosmos()

    vector_store =initialize_vector_store()
    
    relevant_documentation = vector_store.similarity_search(query=question, k=1, search_type="similarity")
    
    context = "\n".join([doc.page_content for doc in relevant_documentation])[:10000]

    #lang = detect(question)

    #print(doc)
    #print(context)
    #print(relevant_documentation)
    source = relevant_documentation[0].metadata['source']
    real_page_no = relevant_documentation[0].metadata['page']
    website_url = relevant_documentation[0].metadata['website_url']

    retriever = vector_store.as_retriever()

    # add chat memory
    memory = ConversationBufferMemory(
        llm = llm,
        output_key='answer',
        memory_key='chat_history',
        chat_memory=history,
        return_messages=True
    )

    #QA_CHAIN_PROMPT = generate_prompt_with_history()

    # The chain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=True,
    )

    #QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    output = chain({"question": question})
    
    logging.info(output)
    answer_value = output["answer"]
    # 
    language = "-"
    
    page_no = ""
    for doc in relevant_documentation:
        page_no = page_no + "," + doc.metadata['page'] 
        
    logging.info(page_no)

    if page_no == "/" or page_no == "N/A":
        source = "-"
        website_url = "/"

    # extract first page number
    if "," in page_no:
        first_page_no  = page_no.split(",")[0]
    else:
        first_page_no = page_no
    
    page_no = extract_page_no(answer_value)
    answer = extract_answer(answer_value)
    
    page_no = int(real_page_no) 
    
    #return answer_value

    #output
    json_response = {
        "raw": answer,
        "answer": answer,
        "source": source,
        "website_url": get_web_url(source),
        "page_no": str(page_no),
        "first_page_no": first_page_no,
        "language": language
    }
    return json_response
