# [Project 1: RetailQueryGPT: Intelligent Inventory Management System](https://github.com/GauBhosale/Inventory-Manager-LLM-Project.git)

## Project Overview
	•	Objective: Create an end-to-end LLM project using Langchain and Google Palm for a retail store selling t-shirts.
	•	Functionality: Develop a tool similar to ChatGPT that converts natural language questions into SQL queries to interact with a MySQL database.
## Key Components
	•	Store Data:
	•	Four main brands: V Hussein, Levi's, Nike, Adidas.
	•	Database has two tables:
	•	t-shirts: Contains inventory count, price per unit.
	•	discounts: Contains discount information per t-shirt.
	•	Personas:
	•	Tony Sharma: Store manager, needs quick answers from the database.
	•	Lok: Data analyst, writes SQL queries but is often busy.
	•	Peter P: Data scientist, tasked with building the LLM tool.
## Technical Architecture
	1.	Question to SQL Query:
	•	Use Google Palm via Langchain to convert questions into SQL queries.
	•	Handle simple to moderately complex queries.
	•	Implement few-shot learning for more complex queries.
	2.	Few-shot Learning:
	•	Prepare a training dataset of sample questions and corresponding SQL queries.
	•	Use embeddings (word/sentence) and store them in a vector database (ChromaDB).
	3.	Vector Database:
	•	Options: Pinecone, Milvus, ChromaDB.
	•	Selected: ChromaDB (open-source and suitable for this project).
	4.	UI Development:
	•	Build using Streamlit.
	•	Minimal coding required (around 5-6 lines).
## Key Steps
	1.	Set Up API Key:
	•	Obtain API key from Google Maker Suite.
	•	Test prompts using Maker Suite's test pad.
	2.	Database Setup:
	•	Use MySQL Workbench to create and manage the database.
	•	Execute provided SQL scripts to set up tables and insert sample data.
	3.	Jupyter Notebook:
	•	Import required libraries and initialize the LLM with the API key.
	•	Test sample prompts and ensure correct functioning.
	4.	SQL Database Chain:
	•	Create an SQL database object in Langchain.
	•	Test queries to ensure accurate SQL generation.
## Challenges and Observations
	•	Complex Queries:
	•	Initial LLM might make errors (e.g., assuming column names).
	•	Few-shot learning needed for handling ambiguities.
	•	Need to explicitly clarify column meanings in the training data.
	•	Examples:
	•	Correct query generation for specific questions (e.g., inventory count, pricing).
	•	Handling of joins and discounts accurately.
## Conclusion
	•	LLMs can significantly streamline database querying processes but require careful setup and training, especially for complex queries.
	•	The integration of vector databases and few-shot learning enhances the tool's capability to handle real-life database scenarios.
This project provides a practical example of using modern LLM frameworks to solve industry-specific problems, demonstrating the potential and challenges of such integrations.


### Final Look 
![](RetailQueryGPT.pdf) 



# [Project 2: Amazon Price Scrapper](https://github.com/RonitMalik/BlackFriday_pythonScrapper)

This was part of a personal project where a python price scrapper was built in order to track prices for specific items on amazon and then send out email alerts. 
The way the model works is you add a link to the amazon product and the price you're willing to buy the product at (Target Price) and then you can run the script and it will refresh the script every 24 hours and check the price for the product, as soon as the price of the product reaches your target price the scrapper will send you an email alert. 

I have also done an entire walkthrough video on youtube for the amazon price scapper and how you can build one too. [Click Here For Youtube Video](https://www.youtube.com/watch?v=vO668yAX3p8)

# [Project 3: Yahoo Finance Web Scrapping](https://github.com/RonitMalik/BlackFriday_pythonScrapper)

This project was part of my youtube channel where i build a yahoo finance web-scrapper to get stock prices from yahoo finance, I used the Yahoo_fin package to get the prices for various stock prices and ran further trend analysis. The main goal of this project was to explore the yahoo_fin package. 

The following video for this project can be found on my [Youtube Channel](https://www.youtube.com/watch?v=AsxpHMq2auc&t=656s)
