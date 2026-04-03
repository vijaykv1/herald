"""RAG based context manager for Herald.

This module implements a context manager that retrieves relevant information to embeddings.
"""

import tqdm
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from agents.tool import function_tool, FunctionTool


class CVVectorStore:
    """A simple vector store implementation for storing and retrieving CV information."""

    def __init__(self, cv_chunks: list):
        """Initialize the vector store.

        :param list cv_chunks: The chunked CV data to be stored in the vector store.
        """
        self.__cv_chunks = cv_chunks
        # In-memory ChromaDB collection — rebuilt on every startup.
        # The CV is small enough that re-embedding takes only a few seconds and
        # avoids any dependency on a persistent filesystem (required for Railway).
        # Uses ChromaDB's built-in ONNX embedding function — no external API needed.
        self.__cv_collection = chromadb.Client().create_collection(
            name="cv_lookup",
            embedding_function=DefaultEmbeddingFunction(),
        )

    def __normalize_chunk(self, chunk: dict) -> str:
        """Normalize the text for better retrieval."""
        topic = chunk.get("topic", "Misc")
        content = ""
        if isinstance(chunk["content"], str):
            content = chunk["content"]
        else:  # dict
            content = "\n".join([f"{k}: {v}" for k, v in chunk["content"].items()])

        # Tag current roles explicitly so "currently working" queries match strongly
        is_current_role = (
            topic == "Experience"
            and isinstance(chunk["content"], dict)
            and "present" in chunk["content"].get("duration", "").lower()
        )
        current_label = "Current Role (Present Position)\n" if is_current_role else ""

        norm_chunk = f"""
### CV Section: {topic}

{current_label}{content}
""".strip()
        # print(f"Normalized chunk:\n{norm_chunk}\n")
        return norm_chunk

    def vectorize_chunks(self):
        """Vectorize the CV chunks and store them in the vector store."""
        for idx, chunk in enumerate(tqdm.tqdm(self.__cv_chunks, desc="Vectorizing CV chunks", colour="green")):
            # normalize the chunk text
            normalized_text = self.__normalize_chunk(chunk)

            # TODO: clean the text if needed (e.g., remove extra whitespace, special characters, etc.)
            # ChromaDB's DefaultEmbeddingFunction handles embedding locally via ONNX — no external API needed.
            self.__cv_collection.add(
                documents=[normalized_text],
                ids=[f"chunk_{idx}"],
                metadatas=[{"topic": chunk.get("topic", "Misc")}],
            )

    def retrieve_relevant_chunks(self, query: str, top_k: int = 4, topic: str = None) -> list:
        """
        Retrieve relevant chunks from the vector store based on the query using cosine similarity search.

        :param str query: The query string to search for relevant CV chunks.
        :param int top_k: The number of top relevant chunks to retrieve.
        :param str topic: Optional topic filter to restrict search to a specific CV section.
        :return: A list of relevant CV chunk texts.
        :rtype: list
        """
        # ChromaDB embeds the query locally and runs cosine similarity search
        query_kwargs = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if topic:
            query_kwargs["where"] = {"topic": topic}

        results = self.__cv_collection.query(**query_kwargs)

        docs = results.get("documents", [])  # get the documents from the results, default to empty list if not found

        return docs[0] if docs else []

    def get_all_chunks_by_topic(self, topic: str) -> list:
        """Return all stored chunks for a given topic without similarity search.

        :param str topic: The topic to filter by (e.g. "Experience").
        :return: All chunk documents for that topic.
        :rtype: list
        """
        results = self.__cv_collection.get(where={"topic": topic})
        return results.get("documents", [])

    def create_tools(self) -> list:
        """Create topic-specific tool wrappers for the retrieve_relevant_chunks method."""

        # FunctionTool used directly so we can supply an explicit schema with
        # "properties": {} — function_tool() on a no-arg function omits "properties",
        # which Groq rejects as invalid JSON Schema.
        async def _list_all_experience_impl(_ctx, _args: str) -> list:
            return self.get_all_chunks_by_topic("Experience")

        list_all_experience_chunks = FunctionTool(
            name="list_all_experience_chunks",
            description=(
                "Return every work experience entry from the CV without any filtering. "
                "Use this tool when the question asks for a complete list — e.g. "
                "'Which companies have you worked at?', 'List all your jobs', "
                "'How many roles have you had?', or any question that requires "
                "enumerating all experience rather than finding the most relevant one."
            ),
            params_json_schema={"type": "object", "properties": {}},
            on_invoke_tool=_list_all_experience_impl,
            strict_json_schema=False,
        )

        @function_tool
        def retrieve_experience_chunks(query: str, top_k: int = 4) -> list:
            """
            Retrieve relevant chunks from the work experience section of the CV.
            Use this tool for questions about jobs, roles, companies, employment history,
            or anything related to where the candidate has worked.

            Args:
                query: The query string to search for relevant experience chunks.
                top_k: The number of top relevant chunks to retrieve (default: 4).

            Returns:
                A list of relevant work experience chunk texts.
            """
            return self.retrieve_relevant_chunks(query, top_k, topic="Experience")

        @function_tool
        def retrieve_skills_chunks(query: str, top_k: int = 3) -> list:
            """
            Retrieve relevant chunks from the skills section of the CV.
            Use this tool for questions about technical skills, programming languages,
            frameworks, tools, or technologies.

            Args:
                query: The query string to search for relevant skills chunks.
                top_k: The number of top relevant chunks to retrieve (default: 3).

            Returns:
                A list of relevant skills chunk texts.
            """
            return self.retrieve_relevant_chunks(query, top_k, topic="Skills")

        @function_tool
        def retrieve_education_chunks(query: str, top_k: int = 3) -> list:
            """
            Retrieve relevant chunks from the education section of the CV.
            Use this tool for questions about degrees, universities, certifications,
            courses, or academic background.

            Args:
                query: The query string to search for relevant education chunks.
                top_k: The number of top relevant chunks to retrieve (default: 3).

            Returns:
                A list of relevant education chunk texts.
            """
            return self.retrieve_relevant_chunks(query, top_k, topic="Education")

        @function_tool
        def retrieve_projects_chunks(query: str, top_k: int = 3) -> list:
            """
            Retrieve relevant chunks from the projects section of the CV.
            Use this tool for questions about personal projects, side projects,
            open source contributions, or specific project details.

            Args:
                query: The query string to search for relevant project chunks.
                top_k: The number of top relevant chunks to retrieve (default: 3).

            Returns:
                A list of relevant project chunk texts.
            """
            return self.retrieve_relevant_chunks(query, top_k, topic="Projects")

        @function_tool
        def retrieve_profile_chunks(query: str, top_k: int = 3) -> list:
            """
            Retrieve relevant chunks from the general profile sections of the CV,
            including summary, contact information, certifications, languages, and publications.
            Use this tool for broad questions about the candidate's overall profile,
            or when no other specific tool applies.

            Args:
                query: The query string to search for relevant profile chunks.
                top_k: The number of top relevant chunks to retrieve (default: 3).

            Returns:
                A list of relevant profile chunk texts.
            """
            return self.retrieve_relevant_chunks(query, top_k)

        return [
            list_all_experience_chunks,
            retrieve_experience_chunks,
            retrieve_skills_chunks,
            retrieve_education_chunks,
            retrieve_projects_chunks,
            retrieve_profile_chunks,
        ]
