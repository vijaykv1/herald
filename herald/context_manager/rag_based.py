"""RAG based context manager for Herald.

This module implements a context manager that retrieves relevant information to embeddings
"""

from herald import cv_parser
from herald.context_manager.icontext import ContextInterface
from herald.cv_parser.linkedin import LinkedInCVParser


class HeraldRAGContextManager(ContextInterface):
    """RAG based context manager for Herald."""

    def __init__(self, cv_pdf_file: str, cv_type: str = "linkedin"):
        """Initialize the RAG based context manager.

        :param str cv_pdf_file: The CV PDF file path
        :param CVParserInterface cv_parser: The CV parser interface to parse the CV and create vector store.
        """
        super().__init__(cv_pdf_file=cv_pdf_file)
        if cv_type == "linkedin":
            self.cv_parser = LinkedInCVParser(cv=self._cv_md_content)
        else:
            raise ValueError(f"Unsupported CV type: {cv_type}")

        # Perform parse and get the chunked data ready for vector store creation
        self._chunks = self.cv_parser.parse()

    @property
    def type(self) -> str:
        """Get the type of the Context Interface.

        :return: Type of the Context Interface
        :rtype: str
        """
        return "rag_based"

    def get_system_instructions(self) -> str:
        """Get the System instructions for Heralder Agent.

        This system instructions is designed for RAG based approach where the agent retrieves relevant information from a vector store based on the user's query.

        1. First read the CV and prepare it in the markdown format
        2. Prepare the system prompt for the Agent to efficiently answer the user's question.
        3. Pass the retrieved relevant information from vector store into the final prompt for clarity to Agent

        :return: System prompt for Agent
        :rtype: str
        """
        # read CV content and create vector store
        self.cv_parser.parse_cv()
        return f"""# CV Assistant System Prompt

    You are a helpful assistant that answers questions about a person's professional background and qualifications based on retrieved information from a vector store.

    ## Your Knowledge Base

    You have access to a vector store that contains information extracted from a CV. When a user asks a question, you will retrieve relevant information from this vector store to help formulate your response.

    ## Instructions

    1. **Retrieve relevant information**: When you receive a query, first retrieve relevant information from the vector store based on the content of the CV.

    2. **Answer directly and concisely**: Provide clear, specific answers based solely on the retrieved information.

    3. **Be accurate**: Only share information that is explicitly stated in the retrieved content. If something isn't mentioned, say so honestly (e.g., "That specific skill/experience isn't mentioned.").

    4. **Handle different question types**:
       - For specific facts (years of experience, job titles, education): Provide exact information
       - For skills/technologies: List relevant ones and where they were used
       - For general questions about background or qualifications: Summarize
"""

    def _prepare_vector_store(self):
        """Prepare the vector store from the CV content."""
        # This method would contain logic to convert the parsed CV content into a format suitable for vector storage and retrieval.
        pass
