"""Module to parse LinkedIn generated CVs.

.. note::

    Lets agree that this module only takes a markdown content as text to parse and understand.

"""

import re
from langchain_text_splitters import MarkdownHeaderTextSplitter

from herald.cv_parser.iparser import CVParserInterface


class LinkedInCVParser(CVParserInterface):
    """A parser for LinkedIn generated CVs."""

    def __init__(self, cv: str):
        """Initialize the parser with the CV content.

        :param str cv: The CV content as a string
        """
        super().__init__(cv)
        self.__linkedin_cv_struct = [("###", "misc_topics"), ("##", "main_topics"), ("#", "name")]

        # topics wise content of the CV
        self.__topics = {
            "misc_topics": [
                "Contact",
                "Skills",
                "Certifications",
                "Languages",
            ],
            "main_topics": ["Experience", "Education", "Projects", "Publications", "Summary", "Patents"],
        }
        self._parsed_cv = None

    @property
    def type(self) -> str:
        return "linkedin"

    @property
    def parsed_cv(self) -> dict:
        return self._parsed_cv

    def parse(self) -> dict:
        """Perform parsing of the CV and return the structured data.

        :return: Parsed CV data as a dictionary
        :rtype: dict
        """
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.__linkedin_cv_struct)
        sections = md_splitter.split_text(self._cv)

        chunks = []
        for section in sections:
            metadata = section.metadata
            if "misc_topics" in metadata:
                topic = metadata["misc_topics"]
                if topic in self.__topics["misc_topics"]:
                    chunks.append({"topic": topic, "content": section.page_content})
            elif "name" in metadata:
                if "main_topics" in metadata:
                    topic = metadata["main_topics"]
                    if topic in self.__topics["main_topics"] and topic not in ["Experience"]:
                        chunks.append({"topic": topic, "content": section.page_content})
                    elif topic == "Experience":  # Experience section is a special case, we will handle it separately
                        experience_content = self._parse_experience(section.page_content)
                        chunks.extend(experience_content)
                    # elif topic == "Education": # Education section is a special case, we will handle it separately
                    #     education_content = self.__parse_education(section.page_content)
                    #     chunks.extend(education_content)
            elif "name" in metadata:  # Just in case its just the name then, we have a special case for it
                name = metadata["name"]
                chunks.append({"topic": "name", "content": name})
                chunks.append({"topic": "overall_description", "content": section.page_content})
            else:  # All other unspecified sections can be added to a miscellaneous topic
                chunks.append({"topic": "miscellaneous", "content": section.page_content})

        self._parsed_cv = chunks
        return chunks

    @staticmethod
    def _parse_experience(content: str) -> dict:
        """Parse the experience section of the CV.

        .. note::

            This routine is a bit more complex than the others because the experience section can have multiple jobs and
            each job can have multiple lines of description. So we need to parse it in a way that we can extract the
            title, company, duration and description for each job.


        :param content: The content of the experience section
        :type content: str
        :return: Jobs as a list of dictionaries with keys "title", "company", "duration", "description"
        :rtype: dict
        """
        jobs = {}
        # print("Content to parse for experience section: \n", content)

        # lets create a simple regex to extract the section into different jobs, we will assume that each job starts
        # with a line that has the format "Title at Company (Duration)"
        job_pattern = re.compile(
            r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}\s*-\s*"
            r"(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
            r"Nov|Dec|January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{4}|\d{4})"
            r"\s*\([^)]*\)$"
        )

        # clean lines
        lines = [
            line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("Page ")
        ]

        jobs = []
        i = 0
        while i < len(lines):
            if job_pattern.match(lines[i]):
                company_info = lines[i - 2]
                title_info = lines[i - 1]
                duration_info = lines[i]
                # location = lines[i+1]
                description_info = []
                i += 1  # lets move the pointer to the next line after the location line
                print("----> Found a job line: ", company_info)
                while i < len(lines) and not job_pattern.match(lines[i]):
                    description_info.append(lines[i])
                    i += 1
                jobs.append(
                    {
                        "topic": "Experience",
                        "content": {
                            "company": company_info,
                            "title": title_info,
                            # "location": location,
                            "duration": duration_info,
                            "description": "\n".join(description_info),
                        },
                    }
                )
            else:
                i += 1

        return jobs


# if __name__ == "__main__":

#     import pprint

#     with open("/Users/Q558981/workspace/personal/herald/data/cv.md", mode="r", encoding="utf-8") as f:
#         cv_content = f.read()
#     parser = LinkedInCVParser(cv_content)
#     parsed_cv = parser.parse()

#     pprint.pprint(parsed_cv, indent=2)
