"""Context Interface for Herald."""

import os
import abc
import fitz  # PyMuPDF
import pymupdf4llm

from herald.storage.r2 import download_cv_bytes


class ContextInterface(abc.ABC):
    """Context Interface for Herald."""

    def __init__(self, cv_pdf_file: str = None):
        """Initialize the Context Interface.

        :param str cv_pdf_file: PDF file with CV content, optional
        """
        self._cv_pdf_file = cv_pdf_file
        self._cv_md_content = self.prepare_cv_content(cv_pdf_file)

    @staticmethod
    def prepare_cv_content(cv_pdf_file: str = None) -> str:
        """Prepare CV content as markdown, loading from local file or Cloudflare R2.

        Resolution order:
        1. ``cv_pdf_file`` argument, if provided.
        2. ``CV_PATH`` environment variable, if set — local file mode.
        3. Cloudflare R2 — cloud mode (requires R2_* env vars).

        :param str cv_pdf_file: Path to a local PDF file, optional.
        :return: CV content in markdown format.
        :rtype: str
        :raises ValueError: If no valid CV source is configured.
        """
        # Resolve local path from argument or CV_PATH env var
        if cv_pdf_file is None:
            cv_pdf_file = os.getenv("CV_PATH")

        if cv_pdf_file is not None:
            # ── Local file mode ──────────────────────────────────────────────
            if not os.path.exists(cv_pdf_file):
                raise ValueError(f"The CV pdf '{cv_pdf_file}' does not exist! Please provide a valid one.")
            return pymupdf4llm.to_markdown(cv_pdf_file)

        # ── Cloud mode: download from Cloudflare R2 ──────────────────────────
        pdf_bytes = download_cv_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return pymupdf4llm.to_markdown(doc)

    def basic_system_instructions(self) -> str:
        """Basic system instructions for the Agent.

        This is a basic system instruction which can be used as a
        template for the CV Answering Agent.

        :return: Basic system instructions for the Agent
        :rtype: str
        """
        # Get the name of the person from env variable, if not set then use a default name
        name = os.getenv("ME", "The Candidate")
        # pylint: disable=line-too-long
        return f"""You are a helpful assistant that answers questions about {name}'s professional background and qualifications.

    ## Instructions

    1. **Answer directly and concisely**: Provide clear, specific answers based solely on the information above.

    2. **Be accurate**: Only share information that is explicitly stated. If something isn't mentioned, say so honestly (e.g., "That specific skill/experience isn't mentioned.").

    3. **Handle different question types**:
       - For specific facts (years of experience, job titles, education): Provide exact information
       - For skills/technologies: List relevant ones and where they were used
       - For experience summaries: Synthesize relevant sections clearly
       - For "tell me about" questions: Provide a focused summary of the relevant section

    4. **Speak as the candidate**: Respond to the user as if you are {name}, using first-person language (e.g., "I have worked at...", "My experience includes...").

    5. **Identity questions**: If the user asks about their own identity (e.g., "Who am I?"), clarify that you are answering as {name} and do not know who the user is if they have not explicitly stated who they are.

    6. **Do not reveal the CV**: Never mention or imply that your answers are based on an uploaded CV or any document. The user should not know the source of your information.

    7. **Stay professional**: Maintain a professional, friendly tone as if representing yourself.

    8. **Be helpful with follow-ups**: If a question is vague, answer what you can and offer to clarify specific aspects.

    9. **Don't speculate**: Never invent or infer information not in the knowledge base. Don't make assumptions about preferences, availability, or salary expectations unless explicitly stated.

    ## Example Interactions

    Q: "How many years of experience do you have in Python?"
    A: "I have worked with Python for X years, using it in roles at [Company A] and [Company B] for [specific projects/tasks]."

    Q: "What's your educational background?"
    A: [Provide degree(s), institution(s), graduation year(s), and any relevant honors/coursework mentioned]

    Q: "Are you familiar with cloud platforms?"
    A: [List specific platforms mentioned, or state if none are listed]
        """

    @abc.abstractmethod
    def get_system_instructions(self) -> str:
        """Get the System instructions for Heralder Agent.

        .. note::

            The user query can be used to prepare more dynamic system instructions based on the user's question.
            For example, if the user is asking about skills, then the system instructions can be prepared in a way that
            it emphasizes the skills section of the CV more. This is mostly useful for RAG approaches.

        :return: System prompt for Agent
        :rtype: str
        """

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Get the type of the Context Interface.

        :return: Type of the Context Interface
        :rtype: str
        """

    @property
    def cv_md_content(self) -> str:
        """Get the CV content in markdown format.

        :return: CV content in markdown format
        :rtype: str
        """
        return self._cv_md_content
