from pydantic import BaseModel, Field
from typing import List

class CorpusDocument(BaseModel):
    cid: str = Field(..., description="ID của văn bản")
    text: str = Field(..., description="Nội dung gốc")
    segmented_text: List[str] = Field(..., description="Danh sách từ đã tách")
    embeddings: List[float] = Field(..., description="Vector embedding của văn bản")