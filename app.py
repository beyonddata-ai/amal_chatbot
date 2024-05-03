#Necessary library to import
import google.generativeai as genai
import google.ai.generativelanguage as glm
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os


from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import gradio as gr 

from google.cloud import translate_v2 as translate
import json
from pathlib import Path

#access keys: llm keys are on keys.json file access those keys before usage
keys_path = Path("./auth/keys.json")
with open(keys_path, 'r') as file:
    api_keys = json.load(file)

#For language support i.e arabic, english
def translate_text(target, text):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './auth/ornate-bebop-415006-4025a20e8711.json'
    translate_client = translate.Client()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)
    return result

#llm=Gemini
# GOOGLE_API_KEY = api_keys["GOOGLE_API_KEY"]
# genai.configure(api_key=GOOGLE_API_KEY)
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
# data_dir = '/media/beyond-data/hdd/credit-rating/chatbot/vector-db/gemini'
# db = FAISS.load_local(data_dir,embeddings)
# retriever = db.as_retriever()

#llm=Gpt3.5 or 4
os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
model = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.3)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
data_dir = './openai-embeddings-small'
db = FAISS.load_local(data_dir,embeddings)
retriever = db.as_retriever()

#Prompt for creating standalone question from chat history and current message
contextualize_q_system_prompt= """Given a chat history and the latest user question, follow these steps:
1. Formulate a standalone question if the question references context in the chat history given below.
2. If the question is not related to the chat history, don't reformulate the question and return it as is.
3. if the question is generic (e.g. hi,hello, thanks, etc), don't reformulate the question and return it as is.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(contextualize_q_system_prompt)

#Prompt for generating answers utilizing context (documents text) and standalone question
template = """
Consider the following for setting the context for next questions: \
You are Amal Bot, and your purpose is to help women entrepreneurs and business owners in their startup journey. \
Consider the user to be an aspiring female entrepreneur without much background knowledge of finance. \
You will also provide them with emotional support. \
You will also provide them with short explanations on business and finance concepts with examples from their own industry or business. \
You will respond with empathy and understand their problem. You advise them on how important it is to manage stress and to build work-life balance. \
You will be their companion so that they won’t feel alone and can access finance and business concepts readily. \
Use second person pronoun “You” to respond. \
In the last two sentences, provide them emotional support with personal touch and  emphasize that they are not alone in this journey. \
Give the concise answer, but don't lose the personal touch. \
Use the context to answer the question at the end. If the provided context lacks information, respond logically from your knowledge base. \
{context}
If the provided context does not contain information, please think rationally and answer from your own knowledge base.
Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

#Combine the retrieved matched documents into one document 
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

#window memory with 5 cells
#memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
memory = ConversationBufferWindowMemory(k=5, output_key="answer", input_key="question", return_messages=True)


# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}

# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}

# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

#main function iinterface to call the final chain
def gemini_chat(user_msg, history):
    translated_input = translate_text(target='en', text=user_msg)
    source_language = translated_input['detectedSourceLanguage']
    query_translation = translated_input['translatedText']
    print(f'Source language: {source_language}')
    if source_language == 'en':
        inputs = {"question": user_msg}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"].content})
        output_text = result['answer'].content
        return output_text
    elif source_language == 'ar':
        inputs = {"question": query_translation}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"].content})
        output_text = result['answer'].content
        translated_output = translate_text(target='ar', text=output_text)
        return translated_output['translatedText']

#For gradio interface
demo = gr.ChatInterface(fn=gemini_chat)
demo.launch()
#demo.launch(auth=("aawaz", "finantler214"), server_name='173.212.242.18', server_port=8005)