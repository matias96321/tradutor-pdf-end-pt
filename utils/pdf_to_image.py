from fastapi import File
from pdf2image import convert_from_bytes
from typing import List

class PdfToImage:
    def __init__(self):
        return

    def __call__(self,file: File) -> List[str]:
        self.file = file
        try:
            images = convert_from_bytes(pdf_file=self.file, dpi=300)
            return images
        except Exception as e:
            print(f"Erro ao converter PDF para imagem: {e}")
            return []

