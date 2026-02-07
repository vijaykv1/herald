"""File containing all the context classes for the herald package."""
import os
import pymupdf4llm
import dotenv


class PromptOptions:
    """Class containing PromptOptions"""

    @staticmethod
    def prepare_pdf_content(cv_pdf_file: str = None) -> str:
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

        if not os.path.exists(cv_pdf_file):
            raise ValueError(
                f"The CV pdf {cv_pdf_file} does not exist! Please provide a valid one."
            )
        
        return pymupdf4llm.to_markdown(cv_pdf_file)

    @staticmethod
    def get_basic_system_instructions() -> str:
        """
        Get the System instructions for Heralder Agent
    
        :return: System prompt for Agent
        :rtype: str
        """
        return """This is the area we write the system prompt for the current Agent"""



