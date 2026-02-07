"""File containing all the context classes for the herald package."""
import os
import pymupdf4llm


class PromptOptions:
    """Class containing PromptOptions"""

    @staticmethod
    def prepare_pdf_content(cv_pdf_file: str) -> str:
        """
        Prepare CV pdf content for the prompt.
        
        :param str cv_pdf_file: PDF file with CV content
        :return: CV content string
        :rtype: str
        """
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



