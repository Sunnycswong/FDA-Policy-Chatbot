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

# Import Azure OpenAI
from langchain.llms import AzureOpenAI 
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

#import textwrap
import logging

# lingua
from lingua import Language, LanguageDetectorBuilder



# setting up credentials
os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "gptdemosearch" # replace with yours search service name
os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = "PcAZcXbX2hJsxMYExc2SnkMFO0D94p7Zw3Qzeu5WjYAzSeDMuR5O" # replace with your api key
os.environ["AZURE_INDEX_NAME"] = "sino-hr-chatbot"
# end setting up credentials

# retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=10)

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

    #lang = detect(question)

    # change to lingua
    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    lang = detector.detect_language_of(question)


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

def generate_prompt():
    prompt_template_string="""
    Follow exactly these 5 steps:
    1. Read the context below and aggregrate this data
    Context : {context}
    2. Answer the question using only this context
    3. Answer the question in less than 200 words
    4. Please provide your answer in English
    5. Please provide the page number of the pages where your answer are based on at the end of your response
    6. Please provide the page numbers in the following output format: [Page: 1, 2, 3]
    User Question: {question}

    Don't justify your answers. Don't give information not mentioned in the given context
    
    If you don't have any context and are unsure of the answer, reply that you don't know about this topic.

    Please provide your answer in English
    """
    prompt_template = PromptTemplate(template = prompt_template_string, input_variables=["context", "question"])

    return prompt_template


def generate_prompt_chi():
    prompt_template_string="""
    指令：
    1. 你必須只根據以下文字的內容回答提問者的詢問。
    2. 如果不懂得回答或文字沒有資料，請回答“對不起，我不懂得回答這個問題。”
    3. 請以少於200字回答問題。
    4. 請在你的回答後提供你用以回答的文字的頁數。格式範例：[Page: 1, 2, 3]
    5. 請以繁體中文回答。

    #####
    文本：{context}
    #####

    問題：{question}

    請以繁體中文回答
    """
    prompt_template = PromptTemplate(template = prompt_template_string, input_variables=["context", "question"])

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

def llm_pipeline(question):

    # set up index name 
    index_name = os.environ["AZURE_INDEX_NAME"] 

    # retrieve information from Azure Search
    relevant_docs, source, website_url, lang, page_no = azure_search_by_index(question, index_name)

    #print(relevant_docs)

    # generate prompt without example

    '''
    if language == "zh-cn" or language == "zh-tw":
        PROMPT = generate_prompt_chi()
    #if language == "en":
    #english prompt
    #    PROMPT = generate_prompt()
    else:
    #english prompt
        PROMPT = generate_prompt()
    
    '''
    
    # use lingua instead
    if lang == Language.CHINESE:
        PROMPT = generate_prompt_chi()
        language = "chinese"
    else:
        PROMPT = generate_prompt()
        language = "english"

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pwcjay.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "f282a661571f45a0bdfdcd295ac808e7"

    # use AzureChatOpenAI 
    llm = AzureChatOpenAI(deployment_name="gpt-35-16k", temperature=0,
                        openai_api_version="2023-05-15", openai_api_base="https://pwcjay.openai.azure.com/")

    chain = LLMChain(llm=llm, 
                    prompt=PROMPT,
                    #verbose=True
                    )

    output = chain.run({"context": relevant_docs, #"context": relevant_docs, 
        "question": question,
        })

    # wrapped_text = textwrap.fill(output, width=100)
    # print(wrapped_text)

    logging.info(output)

    page_no = extract_page_no(output)
    answer = extract_answer(output)

    answer = answer #+ "\n" + f"[page:{page_no}]"

    # if no page number, then the source should be - and website url should be /

    logging.info(page_no)

    if page_no == "/" or page_no == "N/A":
        source = "-"
        website_url = "/"

    # extract first page number
    if "," in page_no:
        first_page_no  = page_no.split(",")[0]
    else:
        first_page_no = page_no

    #if language == "en":
    #    source = "Title: " + source
    #    page_no = "Page: " + page_no
    #else:
    #    source = "文本来源: " + source
    #    page_no = "页数: " + page_no

    json_response = {
        "raw": output,
        "answer": answer,
        "source": source,
        "website_url": website_url,
        "page_no": page_no,
        "first_page_no": first_page_no,
        "language": language
    }

    return json_response #textwrap.fill(question, width=100), textwrap.fill(output, width=100)
    




