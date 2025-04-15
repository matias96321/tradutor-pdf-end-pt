import logging
import re
import os
import tempfile
from string import ascii_letters
import textwrap
from PIL import Image, ImageDraw, ImageFont
from fastapi import File
from matplotlib import pyplot as plt
from networkx import draw
import numpy as np
from utils import OCREngine
from utils import LayoutAnalyzer
from utils import Translator
from utils import PdfToImage

class TranslatePDF:

    def __init__(self):
        
        self.region_result = {}
        self.processed_images = []

        self.temp_dir = os.path.join(tempfile.gettempdir(), "translated_images")
        
        # Garantir que a pasta temporária exista
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize layout analyzer
        self.layout_analyzer = LayoutAnalyzer().analyze

        # Initialize OCR 
        self.ocr_engine = OCREngine()

        # Initialize Translator 
        self.translator_engine = Translator()     

        self.pdf_to_image = PdfToImage()

        self.logger.info("TranslatePDF initialized successfully")
        self.logger.info(f"Temporary directory for images: {self.temp_dir}")
    
    def translate_document(self, file: File) -> str:  
        pages_images = self.pdf_to_image(file)
        processed_images = []

        for i, page_image in enumerate(pages_images):
            try:
                page_image = np.array(page_image)
                self.logger.info(f"Processing page {i+1}/{len(pages_images)}")
                
                layout_results = self.layout_analyzer(page_image)
                self.process_page(page_image, layout_results, i)
                
                page_image = Image.fromarray(page_image)
                processed_images.append(page_image)

            except Exception as e:
                self.logger.error(f"Error processing page {i+1}: {str(e)}")

        pdf_path = self.save_pdf(processed_images) if processed_images else None

        return pdf_path

    def save_pdf(self, processed_images):
        pdf_path = os.path.join(self.temp_dir, "translated_document.pdf")
        # images = [Image.open(img) for img in processed_images]
        if processed_images:
            processed_images[0].save(pdf_path, "PDF", resolution=300.0, save_all=True, append_images=processed_images[1:])
            self.logger.info(f"Saved complete PDF to {pdf_path}")
        return pdf_path

    def process_page(self, page_image, layout_results, page_index):
        for region in layout_results:
            try:
                self.process_region(page_image, region)
            except Exception as e:
                self.logger.warning(f"Skipping region {region.bbox} due to error: {str(e)}")

    def process_region(self, page_image, region):
        x1, y1, x2, y2 = region.bbox
        region_image = page_image[y1:y2, x1:x2]

        ocr_result = self.ocr_engine.extract_text(region_image)
        boxes, rec_res, _ = ocr_result
        text =  ' '.join(result[0] for result in rec_res)
        if text == '':
            return
        print(text)
        print("+++++++++++++++++++++++++++++++++++++++++++")
        text_translated = self.translator_engine(text)
        self.region_result.update({
            "type": region.type,
            "bbox": region.bbox,
            "box_dimen": [x2 - x1, y2 - y1],
            "text": text_translated
        })

        text, font = self.fit_text(text_translated)
        box_img = self.build_bounding_box()
        drawn_img = self.draw_text_in_box(text=text, font=font, box_img=box_img)

        page_image[y1:y2, x1:x2] = drawn_img

    def build_bounding_box(self):
        box_width, box_height = self.region_result["box_dimen"]
        box_img = Image.new('RGB', (box_width, box_height), color=(255, 255, 255))
        return box_img

    def draw_text_in_box(self, text: str, font: ImageFont, box_img: Image) -> Image:
        draw = ImageDraw.Draw(box_img)
        draw.text((0, 0), text, font=font, fill=(0, 0, 0))
        box_img = np.array(box_img)
        return box_img

    def fit_text(self, text: str) -> tuple:
        box_width, box_height = self.region_result["box_dimen"]
        text, point_size = self.get_font_size(textarea=[box_width, box_height], text=text, font_name="./font/SourceHanSerif-Light.otf")
        font = ImageFont.truetype("./font/SourceHanSerif-Light.otf", size=point_size)
        return text, font

    def get_font_size(
        self,
        textarea: list,
        text: str,
        font_name: str,
        pixel_gap: int = 11,
        min_font_size: int = 28,
        max_font_size: int = 90
    ):
        """
        Calcula o tamanho ideal da fonte para que o texto caiba na área especificada.
        
        Args:
            textarea: Lista contendo [largura, altura] da área de texto em pixels
            text: Texto a ser renderizado
            font_name: Caminho para o arquivo de fonte
            pixel_gap: Espaçamento vertical entre linhas em pixels
            min_font_size: Tamanho mínimo da fonte em pontos
            max_font_size: Tamanho máximo da fonte em pontos
            
        Returns:
            Tupla contendo (texto_formatado, tamanho_da_fonte)
        """
        # Verifica se textarea tem os valores necessários
        if len(textarea) < 2:
            raise ValueError("textarea deve conter largura e altura")
            
        text_width = int(textarea[0])
        text_height = int(textarea[1])
        
        original_text = text  # Preserva o texto original
        best_size = min_font_size
        best_wrapped_text = text
        
        # Usa busca binária para encontrar o tamanho ideal mais rapidamente
        low = min_font_size
        high = max_font_size
        
        while low <= high:
            point_size = (low + high) // 2
            
            font = ImageFont.truetype(font_name, point_size)
            ascent, descent = font.getmetrics()
            
            # Calcula a largura média dos caracteres
            avg_char_width = sum(font.getbbox(char)[2] for char in ascii_letters) / len(ascii_letters)
            
            max_char_height = ascent + descent
            
            # Fator de ajuste para a largura (documentado)
            width_adjustment_factor = 1.22  # Ajuste para compensar espaços e variações de largura
            
            max_char_count = int((text_width * width_adjustment_factor) / avg_char_width)
            wrapped_text = textwrap.fill(
                text=original_text,  # Usa o texto original
                width=max_char_count,
            )
            
            num_lines = len(wrapped_text.splitlines())
            total_text_height = (max_char_height * num_lines) + (pixel_gap * (num_lines - 1))
            
            if total_text_height <= text_height:
                # Este tamanho cabe, tente um maior
                best_size = point_size
                best_wrapped_text = wrapped_text
                low = point_size + 1
            else:
                # Este tamanho não cabe, tente um menor
                high = point_size - 1
        
        return best_wrapped_text, best_size
