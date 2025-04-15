#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Document Processing REST API
"""

import os
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from src.translate_pdf import TranslatePDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for document layout detection and OCR processing",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory for results

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/file/upload')
async def _file_upload( file: UploadFile):
    """
    Process multiple documents in batch.
    
    - **files**: List of document files to process
    - **options**: Processing options (optional)
    """
    pdf_file =  file.file.read()
    translator  = TranslatePDF()

    print(pdf_file)
    pdf_file = translator.translate_document(pdf_file)
    
    if not pdf_file:
        raise HTTPException(status_code=400, detail="No files provided")
        
    return FileResponse(pdf_file, media_type='application/pdf')

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
