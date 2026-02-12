"""RAG based context manager for Herald.

This module implements a context manager that retrieves relevant information to embeddings.
"""

import chromadb
from openai import OpenAI


class CVVectorStore:
    """A simple vector store implementation for storing and retrieving CV information."""

    def __init__(self, cv_chunks: list, chromadb_local_path: str = "./cv_vector_store"):
        """Initialize the vector store.

        :param list cv_chunks: The chunked CV data to be stored in the vector store.
        """
        self.__cv_chunks = cv_chunks
        self.__client = OpenAI()
        # create a embeddings collection in chromadb for storing the CV chunks
        self.__cv_collection = chromadb.Client(
            settings=chromadb.config.Settings(persist_directory=chromadb_local_path)
        ).create_collection(name="CV")

    def __normalize_chunk(self, chunk: dict) -> str:
        """Normalize the text for better retrieval."""
        topic = chunk.get("topic", "Misc")
        content = ""
        if isinstance(chunk["content"], str):
            content = chunk["content"]
        else:  # dict
            content = "\n".join([f"{k}: {v}" for k, v in chunk["content"].items()])
        return f"""
### CV Section: {topic}

{content}
""".strip()

    def vectorize_chunks(self):
        """Vectorize the CV chunks and store them in the vector store."""
        for idx, chunk in enumerate(self.__cv_chunks):
            # normalize the chunk text
            normalized_text = self.__normalize_chunk(chunk)

            # TODO: clean the text if needed (e.g., remove extra whitespace, special characters, etc.)

            # Create embeddings of the current chunk using OpenAI embeddings API
            embedding = self.__client.embeddings.create(input=normalized_text, model="text-embedding-3-small")

            # extract the embedding vector from the response
            embedding_vector = embedding.data[0].embedding

            # save to chromadb vector store
            self.__cv_collection.add(
                embeddings=[embedding_vector],
                documents=[normalized_text],
                ids=[f"chunk_{idx}"],
                metadatas=[{"topic": chunk.get("topic", "Misc")}],
            )

    def retrieve_relevant_chunks(self, query: str, top_k: int = 4) -> list:
        """
        Retrieve relevant chunks from the vector store based on the query using cosine similarity search.


        :param str query: The query string to search for relevant CV chunks.
        :param int top_k: The number of top relevant chunks to retrieve.
        :return: A list of relevant CV chunk texts.
        :rtype: list
        """
        # create a embedding for the query
        query_embedding = self.__client.embeddings.create(input=query, model="text-embedding-3-small")

        # extract the embedding vector from the response
        query_embedding_vector = query_embedding.data[0].embedding

        # extract the relavent chunks from the vector store using cosine similarity search
        results = self.__cv_collection.query(
            query_embeddings=[query_embedding_vector],
            n_results=top_k,
        )

        docs = results.get("documents", [])  # get the documents from the results, default to empty list if not found

        return docs[0] if docs else []
