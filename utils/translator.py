from transformers import MarianMTModel, MarianTokenizer

class Translator:

    def __init__(self):

        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def __call__(self, text: str):

        if len(text) < 400:
            return self.translate(text)
        else:    
            return self.translate_large_text(text)
    
    def translate(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def chunk_text(self,text, max_tokens=450):
        sentences = text.split(". ")
        chunks, chunk = [], ""

        for sentence in sentences:
            if len(chunk) + len(sentence) < max_tokens:
                chunk += sentence + ". " 
            else:
                chunks.append(chunk.strip())
                chunk = sentence + ". "

        if chunk:
            chunks.append(chunk.strip())

        return chunks
    
    def translate_large_text(self,text):
        chunks = self.chunk_text(text) 
        translations = [self.translate(chunk) for chunk in chunks]
        return " ".join(translations)