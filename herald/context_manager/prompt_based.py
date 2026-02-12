"""File containing all the context classes for the herald package."""

import os

from herald.context_manager.icontext import ContextInterface


class HeraldBasicPrompter(ContextInterface):
    """Herald Prompt options."""

    def __init__(self, cv_pdf_file: str = None):
        """Initialize the Herald Prompt options.

        :param str cv_pdf_file: PDF file with CV content, optional
        """
        super().__init__(cv_pdf_file=cv_pdf_file)

    @property
    def type(self) -> str:
        """Get the type of the Context Interface.

        :return: Type of the Context Interface
        :rtype: str
        """
        return "basic_prompt"

    def get_system_instructions(self, user_query: str = None) -> str:
        """Get the System instructions for Heralder Agent.

        This system instructions is rudimentary and works in the following way.

        1. First read the CV and prepare it in the markdown format
        2. Prepare the system prompt for the Agent to efficiently answer the user's question.
        3. Pass the CV content into the final prompt for clarity to Agent

        :param str user_query: The user query for which the system instructions is being prepared, not used here
        :return: System prompt for Agent
        :rtype: str
        """
        # read CV content
        cv_content = self._cv_md_content

        cv_instructions = f"""

## Your Knowledge Base

{cv_content}

"""
        final_prompt = self.basic_system_instructions() + cv_instructions
        return final_prompt


# if __name__ == '__main__':

#     import dotenv

#     # load complete dotenv here
#     dotenv.load_dotenv()

#     p_cl = HeraldBasicPrompter()
#     prompt = p_cl.get_system_instructions()
#     print(prompt)
