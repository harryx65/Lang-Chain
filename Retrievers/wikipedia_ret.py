from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2,)

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

docs = retriever.invoke(query)
print(docs)
