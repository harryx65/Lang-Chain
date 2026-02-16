from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableSequence
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

loader = PyPDFLoader(
    'E:\\PureLogics\\python_prac\\Langchain_models\\text_splitters\\Signed Letter Offer.pdf')
doc = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "
)

splitted_text = splitter.split_documents(doc)


prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

chain = prompt | model | parser

results = chain.invoke({'question': "tell me the important points from pdf",
                        'text': splitted_text[1].page_content})

print(results)

# print(len(doc))
