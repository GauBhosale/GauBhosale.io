# [Inventory Manager LLM Project](https://github.com/GauBhosale/Inventory-Manager-LLM-Project.git)

## Project Overview
- **Objective**: Create an end-to-end LLM project using Langchain and Google Palm for a retail store selling t-shirts.
- **Functionality**: Develop a tool similar to ChatGPT that converts natural language questions into SQL queries to interact with a MySQL database.

## Key Components
- **Store Data**:
  - Four main brands: V Hussein, Levi's, Nike, Adidas.
  - Database has two tables:
    - `t-shirts`: Contains inventory count, price per unit.
    - `discounts`: Contains discount information per t-shirt.
- **Personas**:
  - Tony Sharma: Store manager, needs quick answers from the database.
  - Lok: Data analyst, writes SQL queries but is often busy.
  - Peter P: Data scientist, tasked with building the LLM tool.

## Technical Architecture
1. **Question to SQL Query**:
   - Use Google Palm via Langchain to convert questions into SQL queries.
   - Handle simple to moderately complex queries.
   - Implement few-shot learning for more complex queries.
2. **Few-shot Learning**:
   - Prepare a training dataset of sample questions and corresponding SQL queries.
   - Use embeddings (word/sentence) and store them in a vector database (ChromaDB).
3. **Vector Database**:
   - Options: Pinecone, Milvus, ChromaDB.
   - Selected: ChromaDB (open-source and suitable for this project).
4. **UI Development**:
   - Build using Streamlit.
   - Minimal coding required (around 5-6 lines).

## Key Steps
1. **Set Up API Key**:
   - Obtain API key from Google Maker Suite.
   - Test prompts using Maker Suite's test pad.
2. **Database Setup**:
   - Use MySQL Workbench to create and manage the database.
   - Execute provided SQL scripts to set up tables and insert sample data.
3. **Jupyter Notebook**:
   - Import required libraries and initialize the LLM with the API key.
   - Test sample prompts and ensure correct functioning.
4. **SQL Database Chain**:
   - Create an SQL database object in Langchain.
   - Test queries to ensure accurate SQL generation.

## Challenges and Observations
- **Complex Queries**:
  - Initial LLM might make errors (e.g., assuming column names).
  - Few-shot learning needed for handling ambiguities.
  - Need to explicitly clarify column meanings in the training data.
- **Examples**:
  - Correct query generation for specific questions (e.g., inventory count, pricing).
  - Handling of joins and discounts accurately.

## Conclusion
- LLMs can significantly streamline database querying processes but require careful setup and training, especially for complex queries.
- The integration of vector databases and few-shot learning enhances the tool's capability to handle real-life database scenarios.
- This project provides a practical example of using modern LLM frameworks to solve industry-specific problems, demonstrating the potential and challenges of such integrations.

### Final Look 
![](RetailQueryGPT.pdf)

---

# [Loan Eligibility App Notes 📋](https://github.com/GauBhosale/Loan-Eligiblity)

## Overview
- **Backend**: Google Colab
- **Frontend**: Streamlit

## Why Streamlit?
- Converts data scripts into shareable web applications.
- Entirely in Python, no front-end experience required.
- Open-source and free to use.

## Building & Deploying the App
1. **Steps Without Streamlit**
   - Build model
   - Convert to Python script
   - Create Flask app
   - Create frontend with JavaScript
   - Deploy model
2. **Steps With Streamlit**
   - Build model
   - Convert to Python script
   - Create frontend with Streamlit
   - Deploy app

## Rule-Based Model
- **Basic Rule**: If monthly income > ₹50,000, loan is approved; else, rejected.
- **Improved Rule**:
  - If monthly income > ₹50,000, loan approved.
  - If loan amount < ₹5 lakhs, loan approved.
  - Otherwise, loan rejected.

## App Implementation
- **Frontend (Streamlit)**
  - Install libraries: `pyngrok`, `streamlit==0.76.0`
  - Create frontend using Streamlit components:
    - Dropdowns for gender and marital status
    - Numeric inputs for income and loan amount
    - Button to check eligibility
  - Use `st.markdown` for headers, `st.selectbox` for dropdowns, `st.number_input` for numeric inputs.
- **Prediction Function**
  - Takes user input (gender, marital status, income, loan amount)
  - Applies rule-based model to determine loan eligibility
- **Deployment**
  - Use `streamlit run app.py` to host locally on port 8501
  - Use `pyngrok` to make the app publicly accessible

## Example
- **Scenario**: Female, married, income ₹70,000, loan ₹6 lakhs
- **Outcome**: Loan approved (income > ₹50,000)

## Future Topics
- Deploying a machine learning model using Streamlit.
- Hosting models on personal websites.

---

# [Cardiac Arrest Predictor Notes](https://github.com/GauBhosale/Cardiac-Arrest-Predictor.git)

## Problem Statement
- **Objective**: Build a solution to predict the chances of cardiac arrest based on physical and demographic features using machine learning.

## Data
- **Features**: Gender, height, weight, smoking status, alcohol consumption.
- **Target Variable**: cardio (indicating the occurrence of cardiac arrest).

## Tools & Setup
1. **Amazon SageMaker**: For building and deploying models.
2. **Amazon S3**: For storing datasets and trained models.
3. **Libraries**: Numpy, Pandas, Boto3 (AWS SDK), SageMaker, Sklearn.

## Steps Involved
1. **Creating Notebook Instance**
   - Select region and create instance with `ml.t2.medium` type.
   - Assign IAM role for permissions.
2. **Setting up S3 Bucket**
   - Create S3 bucket in the same region as the notebook instance.
3. **Building Model**
   - **Data Loading & Preprocessing**:
     - Load dataset using Pandas.
     - Convert categorical variables to numeric.
     - Split data into training (70%) and testing (30%) sets.
   - **Upload to S3**:
     - Preprocessed data is uploaded to the S3 bucket.
   - **Model Definition**:
     - Use XGBoost algorithm.
     - Define hyperparameters and IAM role.
   - **Training**:
     - Train the model using `ml.m4.xlarge` instance.
     - Store trained model in S3 bucket.
4. **Deploying Model**
   - **Serializer**: Use CSV serializer.
   - **Endpoint Creation**:
     - Deploy model and create an endpoint using `ml.m4.xlarge` instance.
5. **Inference**
   - Convert test set to an array.
   - Generate predictions using the deployed model endpoint.

## Summary
- Preprocess data, train and deploy the model on SageMaker.
- Use endpoint for predictions on new data.
- Steps ensure efficient use of SageMaker and S3 for machine learning workflows.

## Next Steps
- Use the created endpoint to predict cardiac arrest chances for new users in a new Jupyter notebook.

---

# [Typing Tutor Project Notes](https://github.com/GauBhosale/Typing-Tutor.git)

## Overview
- The project is a typing tutor for coders.
- Users type out code displayed on the left side into a coding window on the right.
- Users are shown their typing speed and typing accuracy.

## Key Components
1. **Frontend Creation**:
   - Uses Streamlit for the interface.
   - **Classes Involved**:
     - `TypingTutor`: Manages the frontend and backend.
     - `SessionState`: Manages user sessions.

2. **Class: TypingTutor**
   - **Methods**:
     - `__init__()`: Initializes the website, creates a user session, and loads the deep learning model.
     - `codeGen()`: Generates text using a deep learning model.
     - `getPerf()`: Calculates typing speed and accuracy.
     - `onStartClick()`: Handles the start button click, generates code, and updates session state.
     - `onEvalClick()`: Handles the eval button click, calculates performance, and updates session state.

3. **Class: SessionState**
   - **Variables**:
     - `start_time`: Time when the user starts typing.
     - `end_time`: Time when the user finishes typing.
     - `numchars`: Number of characters in the code.
     - `text`: The code to be typed.
     - `content`: Code typed by the user.

## Workflow
1. **Initialization**:
   - Create an instance of TypingTutor.
   - Calls `__init__()` method:
     - Creates a unique session.
     - Sets up the frontend.
     - Initializes the deep learning model.
2. **User Interaction**:
   - **Start Button Click**:
     - Calls `onStartClick()`:
       - Generates code using `codeGen()`.
       - Updates session state.
       - Displays the generated code on the frontend.
   - **Check Speed Button Click**:
     - Calls `onEvalClick()`:
       - Calculates performance using `getPerf()`.
       - Updates session state.
       - Displays typing speed and accuracy.

## Simplified Frontend Setup
- For learning purposes, replace the deep learning model with a static file `examplecode.py`.
- Focuses on understanding the frontend without backend complexities.

## Implementation Steps
1. **Library Installation**:
   - Install required libraries: `pyngrok`, `streamlit`, `streamlit_ace`.
2. **Project Setup**:
   - Write `examplecode.py`.
   - Create the typing tutor app script.
3. **Running the App**:
   - Use `streamlit run` to start the app.
   - Access the app locally or via ngrok for public access.

## Example Usage
- **Scenario**: User starts typing, the tutor calculates speed and accuracy.
- **Outcome**: Displayed results on the interface.

## Conclusion
- Streamlit provides an easy-to-use framework for creating interactive web applications.
- This project highlights the integration of Streamlit with machine learning models for practical use cases.
