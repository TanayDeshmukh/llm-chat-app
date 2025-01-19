from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class UploadedFileMetadata(BaseModel):
    id: Optional[int] = None
    file_name: str
    file_type: str
    size_on_disk: int
    num_pages: int
    num_chunks: Optional[int] = 0
    num_chunks_error_upload: Optional[int] = 0
    upload_time: Optional[datetime] = None


class VectorDocumentChunk(BaseModel):
    text: str
    page_num: int
    start_idx: int
    paginated_text: bool
    file_name: str
    id: str = Field(default_factory=lambda: str(uuid4()))


class WebDocumentChunk(BaseModel):
    user_query: str
    url: int
    text: str
    summary: int
