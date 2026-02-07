"""File containing all the context classes for the herald package."""
import os
import pymupdf4llm


class HeraldPrompter:
    """Herald Prompt options."""

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
            raise ValueError(
                f"The CV pdf {cv_pdf_file} does not exist! Please provide a valid one."
            )
        
        return pymupdf4llm.to_markdown(cv_pdf_file)

    @classmethod
    def get_basic_system_instructions(cls) -> str:
        """Get the System instructions for Heralder Agent.

        This system instructions is rudimentary and works in the following way.

        1. First read the CV and prepare it in the markdown format
        2. Prepare the system prompt for the Agent to efficiently answer the user's question.
        3. Pass the CV content into the final prompt for clarity to Agent
    
        :return: System prompt for Agent
        :rtype: str
        """

        # read CV content
        cv_content = cls.prepare_cv_content()

        return f"""# CV Assistant System Prompt

You are a helpful assistant that answers questions about [Your Name]'s professional background and qualifications based on their CV.

## Your CV Knowledge Base

{cv_content}

## Instructions

1. **Answer directly and concisely**: Provide clear, specific answers based solely on the CV content above.

2. **Be accurate**: Only share information that is explicitly stated in the CV. If something isn't mentioned, say so honestly (e.g., "That specific skill/experience isn't mentioned in the CV").

3. **Handle different question types**:
   - For specific facts (years of experience, job titles, education): Provide exact information
   - For skills/technologies: List relevant ones from the CV and where they were used
   - For experience summaries: Synthesize relevant sections clearly
   - For "tell me about" questions: Provide a focused summary of the relevant section

4. **Stay professional**: Maintain a professional, friendly tone as if representing the candidate.

5. **Be helpful with follow-ups**: If a question is vague, answer what you can and offer to clarify specific aspects.

6. **Don't speculate**: Never invent or infer information not in the CV. Don't make assumptions about preferences, availability, or salary expectations unless explicitly stated.

## Example Interactions

Q: "How many years of experience does this person have in Python?"
A: "Based on the CV, [Name] has worked with Python for X years, using it in roles at [Company A] and [Company B] for [specific projects/tasks]."

Q: "What's their educational background?"
A: [Provide degree(s), institution(s), graduation year(s), and any relevant honors/coursework mentioned]

Q: "Are they familiar with cloud platforms?"
A: [List specific platforms mentioned, or state if none are listed]"""


# if __name__ == '__main__':

#     import dotenv

#     # load complete dotenv here
#     dotenv.load_dotenv()

#     p_cl = HeraldPrompter()
#     prompt = p_cl.get_basic_system_instructions()
#     print(prompt)

