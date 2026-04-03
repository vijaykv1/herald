"""Module to parse LinkedIn generated CVs.

.. note::

    Lets agree that this module only takes a markdown content as text to parse and understand.

"""

import re
import logging
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
                else:  # Just the name section without any main topics
                    name = metadata["name"]
                    chunks.append({"topic": "name", "content": name})
                    chunks.append({"topic": "overall_description", "content": section.page_content})
            else:  # All other unspecified sections can be added to a miscellaneous topic
                chunks.append({"topic": "miscellaneous", "content": section.page_content})

        self._parsed_cv = chunks
        return chunks

    @staticmethod
    def _build_patterns() -> tuple:
        """Build and return the compiled job date and total-duration regex patterns.

        :return: Tuple of (job_pattern, total_duration_pattern)
        :rtype: tuple
        """
        month_names = (
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
            r"January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
        )
        month_year = rf"{month_names}\s+\d{{4}}"
        start_date = rf"(?:{month_year}|\d{{4}})"
        end_date = rf"(?:Present|{month_year}|\d{{4}})"
        job_pattern = re.compile(
            rf"^{start_date}\s*-\s*{end_date}"
            r"(?:\s*\([^)]*\))?$",
            re.IGNORECASE,
        )
        total_duration_pattern = re.compile(
            r"^\d+\s+years?(?:\s+\d+\s+months?)?$|^\d+\s+months?$",
            re.IGNORECASE,
        )
        return job_pattern, total_duration_pattern

    @staticmethod
    def _resolve_company(lines: list, i: int, total_duration_pattern: re.Pattern, last_company: str) -> str:
        """Resolve the company name for a job entry.

        LinkedIn omits the company name for subsequent roles at the same company, and
        inserts a total-duration line between the company name and the first role title.

        :param list lines: Cleaned lines from the experience section.
        :param int i: Index of the current date line.
        :param re.Pattern total_duration_pattern: Pattern matching total-duration lines.
        :param str last_company: Most recently seen company name.
        :return: Resolved company name.
        :rtype: str
        """
        if i >= 3 and total_duration_pattern.match(lines[i - 2]):
            return lines[i - 3]

        preceding = lines[i - 2]
        is_not_company = (
            preceding.startswith("-")           # bullet point
            or preceding.startswith("•")        # bullet point
            or preceding.endswith(".")          # sentence ending → description
            or len(preceding) > 60              # long line → description
            or (preceding == preceding.lower() and " " not in preceding)  # single lowercase word → artifact
        )
        if last_company and is_not_company:
            return last_company
        return preceding

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
        job_pattern, total_duration_pattern = LinkedInCVParser._build_patterns()

        # clean lines
        lines = [
            line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("Page ")
        ]

        logger = logging.getLogger(__name__)

        jobs = []
        last_company = None
        i = 0
        while i < len(lines):
            if job_pattern.match(lines[i]):
                if i < 2:
                    logger.warning(
                        "Skipping job entry at line %d — not enough preceding lines for company/title extraction.",
                        i,
                    )
                    i += 1
                    continue

                title_info = lines[i - 1]
                company_info = LinkedInCVParser._resolve_company(
                    lines, i, total_duration_pattern, last_company
                )
                last_company = company_info

                duration_info = lines[i]
                description_info = []
                i += 1  # move past the date line
                while i < len(lines) and not job_pattern.match(lines[i]):
                    description_info.append(lines[i])
                    i += 1
                jobs.append(
                    {
                        "topic": "Experience",
                        "content": {
                            "company": company_info,
                            "title": title_info,
                            "duration": duration_info,
                            "description": "\n".join(description_info),
                        },
                    }
                )
            else:
                i += 1

        return jobs


if __name__ == "__main__":

    import pprint

    with open("/Users/Q558981/workspace/personal/herald/data/cv.md", mode="r", encoding="utf-8") as f:
        cv_content = f.read()
    parser = LinkedInCVParser(cv_content)
    parsed_cv = parser.parse()

    pprint.pprint(parsed_cv, indent=2)
