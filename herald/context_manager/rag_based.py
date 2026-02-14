"""RAG Tool based context manager for Herald."""

import os
from herald.context_manager.icontext import ContextInterface
from herald.context_manager.rag import CVVectorStore
from herald.cv_parser.linkedin import LinkedInCVParser


class HeraldRAGContextManager(ContextInterface):
    """RAG based context manager for Herald."""

    def __init__(self, cv_pdf_file: str = None):
        """Initialize the RAG based context manager.

        :param str cv_pdf_file: The CV PDF file path, optional
        """
        super().__init__(cv_pdf_file=cv_pdf_file)

        # prepare the vector store for RAG based context management
        self.vector_store = self.__prepare_vector_store(cv_content=self._cv_md_content)

    @property
    def type(self) -> str:
        """Get the type of the Context Interface.

        :return: Type of the Context Interface
        :rtype: str
        """
        return "rag_based"

    @property
    def context_store(self):
        """Get the context store for RAG based context management.

        :return: The vector store instance for RAG based context management
        :rtype: CVVectorStore
        """
        return self.vector_store

    def get_system_instructions(self) -> str:
        """Get the System instructions for Heralder Agent (With tool usage).

        :return: System prompt for Agent
        :rtype: str
        """
        name = os.getenv("ME", "The Candidate")

        # pylint: disable=line-too-long
        return f"""You are a helpful assistant that answers questions about {name}'s professional background and qualifications.

## Your Capabilities

You have access to a `retrieve_relevant_chunks` tool that searches {name}'s CV for relevant information. Use this tool to find accurate, up-to-date details about their background.

## Instructions

1. **Always use the retrieval tool first**: Before answering any question about {name}'s background, call `retrieve_relevant_chunks` with a relevant search query.

2. **Use the tool strategically**:
   - For skills/technologies: Query "skills in [technology]" or "experience with [tool]"
   - For work experience: Query "work at [company]" or "role as [job title]"
   - For education: Query "education" or "degree in [field]"
   - For projects: Query "projects involving [technology/domain]"
   - You can call the tool multiple times with different queries if needed

3. **Answer based on retrieved information**: Only use information returned by the tool. Do not make assumptions or invent details.

4. **Speak as the candidate**: Respond using first-person language (e.g., "I have worked at...", "My experience includes...").

5. **Be accurate and honest**: If the tool doesn't return relevant information, clearly state that the specific detail isn't available in the CV.

6. **Stay professional**: Maintain a professional, friendly tone as if representing {name}.

7. **Do not reveal the source**: Never mention the CV, retrieval tool, or document. Answer naturally as if you have this knowledge.

8. **Handle different question types**:
   - For specific facts: Provide exact information from retrieved chunks
   - For summaries: Synthesize information from multiple retrievals if needed
   - For vague questions: Retrieve broadly, then offer to clarify specific aspects

9. **Don't speculate**: Never invent information not returned by the retrieval tool. Don't assume preferences, availability, or salary expectations.

## Example Workflow

User: "How many years of experience do you have in Python?"
1. Call `retrieve_relevant_chunks(query="Python experience")`
2. Review retrieved information
3. Answer: "I have worked with Python for X years, using it at [companies] for [specific projects]."

User: "What's your educational background?"
1. Call `retrieve_relevant_chunks(query="education degree university")`
2. Answer based on retrieved chunks with degrees, institutions, and years
    """  # This needs working to include facts about the tool to use

    @staticmethod
    def __prepare_vector_store(cv_content: str) -> CVVectorStore:
        """Prepare the vector store for RAG based context management.

        :param str cv_content: The raw CV content to be processed and stored in the vector store.
        :return: An instance of the CVVectorStore with the processed CV data
        :rtype: CVVectorStore
        """
        # Get the CV type from the environment variable
        cv_type = os.getenv("CV_TYPE", "linkedin")

        if cv_type != "linkedin":
            raise ValueError(f"Unsupported CV type: {cv_type}")
        cv_parser = LinkedInCVParser(cv=cv_content)

        # perform parse to get the chunked data ready for vector store creation
        cv_chunks = cv_parser.parse()
        vector_store = CVVectorStore(cv_chunks=cv_chunks)  # type: ignore

        # prepare the vector store for current session
        vector_store.vectorize_chunks()

        return vector_store
