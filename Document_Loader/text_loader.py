from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableSequence
import os
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate(
    template="""Write EXACTLY 5 lines.
Rules:
- Number each line from 1 to 5
- Each point must be on a new line
- Do NOT write more or fewer than 5 lines

Poem:
{poem}""",
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader(
    'E:\PureLogics\python_prac\Langchain_models\document_loader\cricket.txt', encoding='utf-8')

# loader = PyPDFLoader('dl-curriculum.pdf')
doc = loader.load()

print(doc[0].page_content)

chain = prompt | model | parser

print(chain.invoke({'poem': doc[0].page_content}))
