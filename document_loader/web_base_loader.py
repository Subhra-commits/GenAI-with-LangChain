from langchain_community.document_loaders import WebBaseLoader


url = "https://www.ultimatix.net/content/UtxPortal/us/en/home/megamenu.html"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs)