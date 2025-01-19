import sqlite3
from pathlib import Path
from typing import List

from datetime import datetime

import pandas as pd

from common.constants import DEFAULT_DB_DIR, DEFAULT_DB_NAME
from data.data_classes import UploadedFileMetadata


class UploadedFilesDB:
    def __init__(self, db_dir: Path = DEFAULT_DB_DIR, db_name: str = DEFAULT_DB_NAME):
        self.db_dir = db_dir
        self.db_name = db_name
        self.db_path = self.db_dir / self.db_name
        self.db_dir.mkdir(exist_ok=True)

        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    size_on_disk REAL,
                    num_pages INTEGER NOT NULL,
                    num_chunks INTEGER,
                    num_chunks_error_upload INTEGER,
                    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def insert_file(self, uploaded_file_meta_data: UploadedFileMetadata):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO uploaded_files (file_name, file_type, size_on_disk, num_pages, num_chunks, num_chunks_error_upload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uploaded_file_meta_data.file_name,
                    uploaded_file_meta_data.file_type,
                    uploaded_file_meta_data.size_on_disk,
                    uploaded_file_meta_data.num_pages,
                    uploaded_file_meta_data.num_chunks,
                    uploaded_file_meta_data.num_chunks_error_upload,
                ),
            )

    def read_all_files(self) -> List[UploadedFileMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_name, file_type, size_on_disk, num_pages, num_chunks, num_chunks_error_upload, upload_time
                FROM uploaded_files
                """
            )
            rows = cursor.fetchall()

        files = [
            UploadedFileMetadata(
                id=row[0],
                file_name=row[1],
                file_type=row[2],
                size_on_disk=row[3],
                num_pages=row[4],
                num_chunks=row[5],
                num_chunks_error_upload=row[6],
                upload_time=datetime.strptime(row[7], "%Y-%m-%d %H:%M:%S"),
            )
            for row in rows
        ]

        return files

    def read_all_files_as_df(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_name, file_type, size_on_disk, num_pages, num_chunks, num_chunks_error_upload, upload_time
                FROM uploaded_files
                """
            )
            rows = cursor.fetchall()

        columns = [
            "id",
            "file_name",
            "file_type",
            "size_on_disk",
            "num_pages",
            "num_chunks",
            "num_chunks_error_upload",
            "upload_time",
        ]
        df = pd.DataFrame(data=rows, columns=columns)

        return df

    def file_exists(self, file_name: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, file_name, file_type, size_on_disk, num_pages, num_chunks, num_chunks_error_upload, upload_time
                FROM uploaded_files
                WHERE file_name = ?
                """,
                (file_name,),
            )
            rows = cursor.fetchall()
            return len(rows) > 0
