from pathlib import Path
from typing import List, Tuple

import fitz

from data.data_classes import VectorDocumentChunk


def extract_text_pages_from_pdf(pdf_path: Path) -> List[str]:
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    return pages


# split index is decided based on the nearest period(.) To have a smooth transition between the chunks.
# if no period is found near the split index, then the preference is given to a new line,
# and then a space has the least preference
def find_split_idx(text: str, split_idx: int) -> int:
    def closest_point(point1: int, point2: int, candidate: int):
        return point1 if abs(candidate - point1) < abs(candidate - point2) else point2

    prev_split_candidate_period = text.rfind(".", 0, split_idx)
    next_split_candidate_period = text.find(".", split_idx)

    if prev_split_candidate_period != -1 and next_split_candidate_period != -1:
        return closest_point(
            prev_split_candidate_period, next_split_candidate_period, split_idx
        )

    prev_split_candidate_space = next_split_candidate_space = -1

    if prev_split_candidate_period == -1:
        if next_split_candidate_period != -1:
            return next_split_candidate_period
        prev_split_candidate_space = text.rfind(" ", 0, split_idx)

    if next_split_candidate_period == -1:
        if next_split_candidate_period != -1:
            return prev_split_candidate_period
        next_split_candidate_space = text.find(" ", split_idx)

    if prev_split_candidate_space != -1 and next_split_candidate_space != -1:
        return closest_point(
            prev_split_candidate_space, next_split_candidate_space, split_idx
        )
    return split_idx


# A rolling window approach is used to have smooth transitions between the chunks, even when the chunks span multiple pages.
def rolling_window(
    pages: Tuple[str, str],
    page_numbers: Tuple[int, int],
    file_name: str,
    start_idx: int = 0,
    chunk_size: int = 1000,
    overlap_size: int = 200,
    last_page: bool = False,
) -> tuple[list[VectorDocumentChunk], int]:
    chunks = []
    full_text = "".join(pages)
    while True:
        paginated_text = False

        end_idx = start_idx + chunk_size
        if end_idx > len(full_text):
            return chunks, start_idx - len(pages[0])

        if end_idx < len(pages[0]):
            curr_page_num = page_numbers[0]
        else:
            curr_page_num = page_numbers[1]
            if start_idx < len(pages[0]):
                paginated_text = True
            if last_page:
                end_idx = len(full_text) - 1

        chunk_text = full_text[start_idx:end_idx]
        document = VectorDocumentChunk(
            text=chunk_text,
            page_num=curr_page_num - 1 if paginated_text else curr_page_num,
            start_idx=(
                start_idx if start_idx < len(pages[0]) else start_idx - len(pages[0])
            ),
            paginated_text=paginated_text,
            file_name=file_name,
        )
        chunks.append(document)
        start_idx = find_split_idx(full_text, end_idx - overlap_size)


def extract_document_chunks(
    pages: List[str], file_name: str, chunk_size: int, overlap_ratio: int
) -> List[VectorDocumentChunk]:
    i = 0
    j = i + 1
    start_idx = 0
    all_chunks = []
    overlap_size = int(chunk_size * overlap_ratio)
    while j < len(pages):
        page1 = pages[i]
        page2 = pages[j]
        chunks, start_idx = rolling_window(
            (page1, page2),
            (i, j),
            file_name,
            start_idx,
            chunk_size,
            overlap_size,
            j == len(pages) - 1,
        )
        all_chunks.extend(chunks)
        i += 1
        j = i + 1
    return all_chunks
