# FinGPT-0.5
***SINCE THIS PROJECT IS INSPIRED BY FINGPT PAPER BUT NOT AS GOOD AS FINGPT NAME AS FINGPT-0.5***

A fine-tuned LLaMA model that provides detailed information about NASDAQ-listed stocks on demand.This model is trained on 7k Fianancial dataset to be able to answer your question.

# Features
* Extract Company Overview from Alpha Vantage API.
* Extract Stock data.
* Extract Stock News, Stock balance sheet from Alpha vantage API and more different SEC_api for SEC data.
* Symmarize all data combined for overall stock information.
* Provide brief Company Overview and sector information

# Usage
* Provide model with NASDAQ stock ticker and your Question.
* It then extract data from APIs and send its chunks to vector-database created by Facebook name FAISS(Facebook Artificial Intelligence Similarity Search).
* Make a context and pass question with context to LLM Model pretrained and fine tuned with Financial Dataset.

# Model Details
### Architecture:

* Based on the transformer architecture,Uses standard transformer blocks with pre-normalization,Employs rotary positional embeddings (RoPE)


### Training Approach:

* Trained using optimized transformer implementation
* Uses AdamW optimizer
* Incorporates cosine learning rate schedule
* Employs gradient clipping and weight decay


### Tokenization:

* Uses byte-pair encoding (BPE) with a vocabulary size of 32,000 tokens


### Context Window:

* LLaMA: 2048 tokens
* LLaMA 2: Extended to 4096 tokens


### Training Data:

* Trained on approximately 1.4 trillion tokens
* Data sources include CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, and StackExchange


### Compute Requirements:

* Training of the 65B model used 2048 A100 GPUs for 21 days


### Inference Optimization:

* Supports various quantization techniques for faster inference
* Can run on consumer hardware with appropriate optimizations


### Performance Metrics:

* Evaluated on various benchmarks including MMLU, HellaSwag, TruthfulQA
* Shown to outperform many larger models on certain tasks


### Key Innovations:

* Grouped-query attention for improved efficiency
* Improved fine-tuning techniques in LLaMA 2

### Fine Tuning Details:
* Used Llama -3 8B model
* Used model from famous Unsloth Library makes fine tuning more efficient and faster.
* loaded using 4 bit qunatization for reduced memory usage.
* Use Parameter Efficient Fine Tuning Method with rank 8 alpha 32.
* Total trainable parameters before PEFT was 8,037,076,992 after applying PEFT reduced to 0.0848% of total params means 6,815,744.

# Example of Usage
### Sample Input:
#### Input Contains 2 things:
. NASDAQ Ticker.
. Question related to that Ticker.
#### EX:
symbol = 'AMZN'
question = "Explain the growth rate?"
#### Output:
First we create context from vector database FAISS, Then pass context to the prompt with question and ask to generate Response(which looks like below).\
\
**Revenue and Profit Figures:**

* Net Sales:
	+ 2022: $513,983 million
	+ 2023: $574,785 million
	+ Year-over-year growth: 12%
* North America:
	+ 2022: $315,880 million
	+ 2023: $352,828 million
	+ Year-over-year growth: 12%
* International:
	+ 2022: $118,007 million
	+ 2023: $131,200 million
	+ Year-over-year growth: 11%
* AWS:
	+ 2022: $80,096 million
	+ 2023: $90,757 million
	+ Year-over-year growth: 13%

**Year-over-Year Growth Rate:**

* North America: 12% (2022-2023)
* International: 11% (2022-2023)
* AWS: 13% (2022-2023)
* Consolidated: 12% (2022-2023)

**Debt Levels and Ratios:**
No debt information provided.

**Market Share:**
Not provided.

**Notable Trends or Changes:**
* Sales growth driven by increased unit sales, primarily by third-party sellers, advertising sales, and subscription services.
* Continued focus on price, selection, and convenience for customers, including from shipping offers.
* Changes in foreign exchange rates affected net sales by $71 million in 2023.

**Future Growth in Percentages:**
Not provided.

**Overall Sentiment:**
The company's sales growth is steady, with a 12% increase in 2023 compared to the prior year. The growth is driven by increased unit sales, advertising sales, and subscription services. However, the company's revenue growth may not be sustainable, and the growth rate may decrease in the future. The overall sentiment is neutral, with a slight upward trend.\
# Conclusion
Since this model performs well and also give sentiment analysis of this data.
But still lags with more detailed review of stock.\
Since I am still trying to improve it Further and will update further if Done.
*thanks.*




