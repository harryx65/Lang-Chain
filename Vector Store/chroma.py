from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


doc1 = Document(
    page_content="Fakhar Zaman is one of the most successful and consistent batsmen in PSL history. Known for his aggressive batting style and fitness.",
    metadata={"team": "Lahore Qalandars"}
)
doc2 = Document(
    page_content="Muhammad Rizwan is the most successful captain in PSL history, leading Multan Sultans to 2 titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Multan Sultans"}
)
doc3 = Document(
    page_content="Sarfraz Ahmend, famously known as Captain Cool, has led Quetta Gladiators to multiple PSL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Quetta Gladiators"}
)
doc4 = Document(
    page_content="Shaheen Shah is considered one of the best fast bowlers in T20 cricket. Playing for Lahore Qalandars, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Lahore Qalandars"}
)
doc5 = Document(
    page_content="Muhammad Nawaz is a dynamic all-rounder who contributes with both bat and ball. Representing Karachi Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Karachi Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='E:\\PureLogics\\python_prac\\Langchain_models\\vector_base\\chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

results = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
# print(results)

query_results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=1
)
print(query_results)

filter = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Karachi Kings"}
)
print(filter)
