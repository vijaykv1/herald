"""Context Interface for Herald."""

import os
import abc
import pymupdf4llm


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
        """Prepare CV pdf content for the prompt.

        .. note::
            CV file can be loaded via Environment var ``CV_PATH``

        :param str cv_pdf_file: PDF file with CV content, optional
        :return: CV content string
        :rtype: str
        """
        # if cv_pdf_file is not set, then an attempt is made to read from the env variables
        if cv_pdf_file is None:
            cv_pdf_file = os.getenv("CV_PATH")

        if not os.path.exists(cv_pdf_file) or cv_pdf_file is None:
            raise ValueError(f"The CV pdf {cv_pdf_file} does not exist! Please provide a valid one.")

        return pymupdf4llm.to_markdown(cv_pdf_file)

    @abc.abstractmethod
    def get_system_instructions(self) -> str:
        """Get the System instructions for Heralder Agent.

        This system instructions is rudimentary and works in the following way.

        1. First read the CV and prepare it in the markdown format
        2. Prepare the system prompt for the Agent to efficiently answer the user's question.
        3. Pass the CV content into the final prompt for clarity to Agent

        :return: System prompt for Agent
        :rtype: str
        """
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Get the type of the Context Interface.

        :return: Type of the Context Interface
        :rtype: str
        """
        pass
