import yfinance as yf
import wbdata
from datetime import datetime
from sec_edgar_downloader import Downloader
from datetime import datetime, timedelta
import os
import json
from transformers import AutoTokenizer,AutoModelForCausalLM
from unsloth import FastLanguageModel
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from peft import PeftModel
from model.train import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sec_api import ExtractorApi,QueryApi

class Inference:
    def __init__(self):
        self.dl = Downloader(company_name='Abhay',email_address="Abhaychourasiya945@gmail.com")
        self.sec_api_key = os.getenv("sec_api")
        self.hf = os.getenv("HF_TOKEN")
        self.tokenizer = None
        self.model = None  # Initialize model as None
        self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Summarize the following financial report. Include ALL key statistical data and metrics. Focus on:

                1. Revenue and profit figures.
                2. Year-over-year growth rates.
                3. Profit margins.
                4. Debt levels and ratios.
                5. Market share.
                6. Notable trends or changes.
                7. predict Future Price Gain or Loss in percentage.
                8. Summary should be concise and to the mark with relevant data.
                9. Give Overall sentiment according to data provided with Up and down trend indication.
                10. You are a good financial analyzer and analyze effectively.

                Provide a comprehensive yet concise summary suitable for financial professionals.
                User will give Context and Question according to which assistant have to produce Summary and Sentiment.<|eot_id|>

                <|start_header_id|>user<|end_header_id|>
                ### Question:
                {}
                ### Context:
                {}

                <|eot_id|>
                ### Response:
                <|start_header_id|>assistant<|end_header_id|>
                {}
                <|eot_id|>
                """
        print("Initialized Inference class.")
    def formating(self,example):
        print("Running formating method.")
        contexts = example['context']
        questions = example['question']
        answers = example['answer']
        texts = []
        for context,question,answer in zip(contexts,questions,answers):
            text = self.prompt.format(question,context,answer)
            texts.append(text)
        return {'text':texts,}


    def get_stock_data(self,symbol, period="1mo"):
        print(f"Fetching stock data for {symbol} over {period}.")
        """Extract stock data and convert to text format."""
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)

        text_data = []
        for date, row in data.iterrows():
            text = f"On {date.date()}, {symbol} stock opened at {row['Open']:.2f}, "
            text += f"reached a high of {row['High']:.2f}, a low of {row['Low']:.2f}, "
            text += f"and closed at {row['Close']:.2f}. The trading volume was {row['Volume']}."
            text_data.append(text)

        return "\n".join(text_data)

    def get_alpha_vantage_news(self,symbol):
        print(f"Fetching news data for {symbol}.")
        """Extract news data from Alpha Vantage and convert to text format."""
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey=X4AM5ZR6E2WPLQ7C"
        response = requests.get(url)
        data = response.json()

        if 'feed' not in data:
            return "No news data available."

        text_data = []
        for article in data['feed'][:10]:  # Limit to 10 articles
            text = f"Title: {article['title']}\n"
            text += f"Date: {article['time_published']}\n"
            text += f"Summary: {article['summary']}\n"
            text += f"Sentiment: {article['overall_sentiment_label']} (score: {article['overall_sentiment_score']})\n\n"
            text_data.append(text)

        return "\n".join(text_data)

    
    def get_sec_data(self,ticker,type = '10-K'):
        print(f"Fetching SEC data for {ticker} of type {type}.")
        queryApi = QueryApi(self.sec_api_key)
        query = {
            "query":f"ticker:{ticker} AND formType:\"{type}\"",
            "from": "0",
            "size": "1",
            "sort": [{"filedAt":{"order":"desc"}}]
        }
        filings = queryApi.get_filings(query)
        filing_url = filings["filings"][0]['linkToFilingDetails']

        extractorApi = ExtractorApi(self.sec_api_key)
        onea_text = extractorApi.get_section(filing_url,"1A","text")
        seven_text = extractorApi.get_section(filing_url,"7","text")

        combined_text = onea_text + "\n\n" + seven_text
        return combined_text
    def company_overview(self,symbol):
        print(f"Fetching company overview for {symbol}.")
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey=X4AM5ZR6E2WPLQ7C'
        r = requests.get(url)
        data = r.json()
        text = ""
        for key,value in data.items():
            subtext = f"The {key} is {value}. "
            text += subtext
        return text
    def get_balance_sheet(self,symbol):
        print(f"Fetching balance sheet for {symbol}.")
        url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey=X4AM5ZR6E2WPLQ7C'
        r = requests.get(url)
        data = r.json()
        data = data['annualReports']
        text = ""
        for dict in data:
            for key,value in dict.items():
                subtext = f"The {key} is {value}. "
                text += subtext
        return text
    def get_combine_data(self,symbol):
        print(f"Combining data for {symbol}.")
        overview = self.company_overview(symbol=symbol)
        balance_sheet = self.get_balance_sheet(symbol=symbol)
        stock_data = self.get_stock_data(symbol=symbol)
        news = self.get_alpha_vantage_news(symbol=symbol)
        sec_data = self.get_sec_data(ticker=symbol)

        return overview,balance_sheet,stock_data,news,sec_data
    def load_embeddings(self):
        print("Loading embeddings.")
        model_path = 'BAAI/bge-large-en-v1.5'
        model_kwargs = {"device":"cuda"}
        encode_kwargs = {"normalize_embeddings":True}
        embeddings = HuggingFaceEmbeddings(model_name = model_path,model_kwargs = model_kwargs,encode_kwargs = encode_kwargs)
        return embeddings
    def get_vectorspace(self,symbol,embeddings):
        print(f"Creating vector space for {symbol}.")
        data = """### COMPANY OVERVIEW:
        {}

        ### BALANCE SHEET:
        {}

        ### STOCK DATA:
        {}

        ### STOCK NEWS:
        {}\

        ### SEC_DATA:
        {}"""

        overview,balace_sheet,stock_data,stock_news,sec_data = self.get_combine_data(symbol=symbol)
        data = data.format(overview,balace_sheet,stock_data,stock_news,sec_data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 500,length_function = len,is_separator_regex=False)
        split_data = text_splitter.create_documents([data])

        db = FAISS.from_documents(split_data,embedding=embeddings)
        retriever = db.as_retriever()
        return retriever
    # def get_context(self,symbol,query,embeddings):
    #     print(f"Retrieving context for {symbol} with query: {query}.")
    #     retriever = self.get_vectorspace(symbol=symbol,embeddings = embeddings)
    #     retrieved_doc = retriever.invoke(query)
    #     context = []
    #     for doc in retrieved_doc:
    #         context.append(doc.page_content)
    #     return context

    def inference(self, context, question):
        print("Running inference.")
        # Clear CUDA cache to free up memory
        import torch
        torch.cuda.empty_cache()

        # Combine context and question for a single input
        input_text = self.prompt.format(question, context, '')
        inputs = self.tokenizer(
            [input_text],
            return_tensors='pt'
        )
        
        # Optionally switch to CPU if out of memory
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = inputs.to(device)
        
        # Use no_grad to save memory
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1024, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)  # Reduced max_new_tokens
        response = self.tokenizer.batch_decode(output)
        
        # Clear CUDA cache after inference
        torch.cuda.empty_cache()
        
        return response
    def answer_extract(self,text):
        print("Extracting answer from text.")
        text = text[0]
        start_token = "Response:\n<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>\n"
        end_token = "<|eot_id|>"
        start_idx = text.find(start_token)
        if start_idx == -1:
            return None
        start_idx += len(start_token)
        
        while start_idx < len(text) and text[start_idx] in '\n':
            start_idx += 1
        
        end_idx = text.rfind(end_token)  
        if end_idx == -1:
            return None
        return text[start_idx:end_idx].strip()
    def format_output(self,text):
        print("Formatting output.")
        # Split the text into lines
        lines = text.split('\\n')
        for line in lines:
            print(line)
    
    def load_model(self):
        print("Loading model.")
        self.tokenizer = AutoTokenizer.from_pretrained("Abhay06102003/Llama-3-FinanceAgent",token = self.hf)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=self.hf, load_in_4bit=True)
        model = PeftModel.from_pretrained(model, "Abhay06102003/Llama-3-FinanceAgent")
        self.model = model.merge_and_unload()  # Store the loaded model
    
    def get_context(self,question):
        retrieved_doc = self.retriever.invoke(question)
        context = ""
        for doc in retrieved_doc:
            context += doc.page_content
            context += ' '
        return context

#     def main(self, symbol, question):
#         print(f"Running main with symbol: {symbol} and question: {question}.")
#         embedding = self.load_embeddings()
        
#         # Create and store embeddings in the database
#         self.retriever = self.get_vectorspace(symbol=symbol, embeddings=embedding)
        
#         # Clear CUDA cache to free up memory before loading the LLM
#         import torch
#         torch.cuda.empty_cache()
#         self.load_model()
#         context = self.get_context(question)
#         resp = self.inference(context=context,question=question)
#         return self.format_output(self.answer_extract(resp))
        
# if __name__ == "__main__":
#     symbol = "AMZN"
#     question = "Give Predicted Potential of this company?"
#     inferences = Inference()
#     print(inferences.main(symbol=symbol,question=question))
    
    