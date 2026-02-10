"""Parser interface for CV parsing in herald system."""
import abc


class CVParserInterface(abc.ABC):
    """Abstract base class for CV parsers."""
    
    @abc.abstractmethod
    def parse(self, cv: str) -> dict:
        """Parse the CV and return structured data.

        :param cv: The CV content as a string
        :type cv: str
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
