"""Parser interface for CV parsing in herald system."""

import abc


class CVParserInterface(abc.ABC):
    """Abstract base class for CV parsers."""

    def __init__(self, cv: str):
        """Initialize the parser with the CV content.

        .. note::

            Only accepts a file which is either a markdown or a text file,
            or a string content of the CV itself.
            The parser will determine the type based on the content.

        :param str cv: The CV content as a string
        """
        # Check if cv is a file path or a string content
        if cv.endswith(".txt") or cv.endswith(".md"):
            with open(cv, "r", encoding="utf-8") as file:
                self._cv = file.read()
        else:
            self._cv = cv

    @abc.abstractmethod
    def parse(self) -> dict:
        """Parse the CV and return structured data.

        :return: Parsed CV data as a dictionary
        :rtype: dict
        """
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Return the type of CV this parser handles."""
        pass

    @property
    @abc.abstractmethod
    def parsed_cv(self) -> dict:
        """Return the parsed CV data."""
        pass
