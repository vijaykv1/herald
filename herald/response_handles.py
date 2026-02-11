"""All response handles for the model."""

from pydantic import BaseModel, Field


class CVResponse(BaseModel):
    answer: str = Field(..., description="Answer provided for the user query")
