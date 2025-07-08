#%%
# Imports

import psycopg2
import time
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLLM
from typing import Any, List, Mapping, Optional
import os
from langchain_core.outputs import Generation, LLMResult
# ***** Added/Modified section: Import SQLDatabaseToolkit and SQLDatabase from langchain-community *****
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

#%%
# Connect to PostgreSQL (psycopg2)

conn = psycopg2.connect(
    dbname="d365_database",
    user="postgres",
    password="Csi@dmin9",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

cursor.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
""")

tables = [f'public.{row[0]}' for row in cursor.fetchall()]
print("Tables in public:", tables)

#%%
# Database connection

engine = create_engine(
    "postgresql://postgres:Csi%40dmin9@localhost:5432/d365_database",
    connect_args={"options": "-c search_path=public"}
)

db = SQLDatabase(engine=engine)

table_info = db.get_table_info()
print(table_info)
#%%
# Cuda check

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Running on device: {device}")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"CUDA Version: {torch.version.cuda}")
# print(f"Device Count: {torch.cuda.device_count()}")

#%%
# Load Qwen Tokenizer and Model

# torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")
model = AutoModelForCausalLM.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")
# model = model.to(device)

# llm = Ollama(model="qwen3:4b")
# # Test model
# response = llm("สวัสดีครับ")
# print(response)

#%%
# Define Custom LLM for the Agent (must be defined before it's used by toolkit or agent)
class CustomQwenAgentLLM(BaseLLM):
    """Custom LLM for LangChain agent that uses your Qwen model."""

    tokenizer: Any
    model: Any

    @property
    def _llm_type(self) -> str:
        return "custom_qwen_agent_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        result = self._generate([prompt], stop=stop, **kwargs)
        return result.generations[0][0].text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate text from prompts."""
        generations = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # --- IMPORTANT CHANGE 1: Use max_new_tokens for agent's reasoning ---
            # Set max_new_tokens to be sufficient for a thought, action, and action input.
            # 512 tokens is usually a good starting point for agent steps.
            outputs = self.model.generate(**inputs, max_new_tokens=512, **kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append([Generation(text=generated_text)])

        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": "Ellbendls/Qwen-2.5-3b-Text_to_SQL"}

# Initialize the custom LLM
agent_llm = CustomQwenAgentLLM(tokenizer=tokenizer, model=model)

#%%
# 4. Create SQLDatabaseToolkit (NOW with agent_llm passed)
# This allows the toolkit to use an LLM for its own internal reasoning tasks.
toolkit = SQLDatabaseToolkit(db=db, llm=agent_llm)

#%%
# Fetch all table information from the database
actual_table_info = db.get_table_info()

#%%
# Define PromptTemplate (using actual_table_info)

prompt_template = PromptTemplate(
    input_variables=["input", "dialect", "table_info"],
    template="""
You are an AI model designed to assist the sales team in querying data from an existing PostgreSQL database. 
Your job is to generate SQL queries to retrieve information from the 'public' schema, based on the user's request.

Only generate valid SQL queries to fetch data from the already existing tables in the database. Do not include any extra context, explanations, or additional output like 'Context:', 'Generated SQL:', etc.

Here is the list of tables you can query:
{table_info}

Given the user's question: {input}

Output only the SQL query. No explanations or additional context. Use dialect: {dialect}

Examples:
Q: Total sales for each month in the year 2024
SQL: SELECT TO_CHAR(salesorders.createdon::date, 'YYYY-MM') AS sales_month, SUM(salesorders.totalamount::numeric) AS total_sales FROM salesorders WHERE EXTRACT(YEAR FROM salesorders.createdon::date) = 2024 GROUP BY sales_month ORDER BY sales_month;

Q: For October 2024, find the total sales amount for each customer.
SQL: SELECT a.name AS customer_or_account_name, SUM(so.totalamount::numeric) AS total_sales FROM salesorders so JOIN accounts a ON so._customerid_value = a.accountid WHERE so.createdon::date >= '2024-10-01' AND so.createdon::date < '2024-11-01' GROUP BY a.name ORDER BY total_sales DESC;

Q: List all customer names associated with the salesperson 'Kosith Theingtrong'.
SQL: SELECT c.fullname AS customer_name FROM contacts c JOIN systemusers su ON c._ownerid_value = su.systemuserid WHERE su.fullname = 'Kosith Theingtrong';

Given the user's question: {input}

Output only the SQL query. No explanations or additional context. Use dialect: {dialect}

---SQL_QUERY_START---
"""
)

# Define generate_sql_query_tool (using actual_table_info)
def generate_sql_query_tool(query: str, dialect: str = "postgresql") -> str:
    """
    Generates an SQL query based on a natural language question.
    Useful for answering questions that require querying a PostgreSQL database.
    Input should be a clear natural language question about the database.
    """
    start_time = time.time()
    prompt = prompt_template.format(input=query, dialect=dialect, table_info=actual_table_info)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # --- IMPORTANT CHANGE 2: Use max_new_tokens for SQL generation ---
    # SQL queries can be long, but 500-1000 tokens should be sufficient for most cases.
    # Adjust this value based on the expected complexity/length of your SQL.
    outputs = model.generate(**inputs, max_new_tokens=2000)
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    sql_start_tag = "---SQL_QUERY_START---\n"
    
    if sql_start_tag in full_output:
        sql_query = full_output.split(sql_start_tag, 1)[1].strip()
    else:
        sql_query = full_output.strip()
        print("Warning: SQL start tag was not found in the model's output. Returning full output.")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to generate SQL: {elapsed_time:.2f} seconds")
    
    return sql_query

# Define your custom SQL generator Tool
sql_generator_tool = Tool(
    name="SQL_Generator",
    func=generate_sql_query_tool,
    description="Useful for generating SQL queries from natural language questions about the PostgreSQL database. Input should be a natural language question like 'Show me the total sales for last month'."
)

# Combine all tools for the Agent
tools = toolkit.get_tools() # These include sql_database_query_tool, sql_database_schema_tool, etc.
tools.append(sql_generator_tool) # Add our Qwen-based SQL generator to the list

# ***** Modified section: Create AgentPrompt and AgentExecutor *****
# We could use LangChain's create_sql_agent for a specialized SQL Agent,
# but since we want to specifically use Qwen-2.5-3b-Text_to_SQL as the primary SQL generation model,
# we'll stick with create_react_agent and our custom prompt_template for more control.


agent_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant that can generate SQL queries and interact with databases.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Example Usage of the Agent
user_query = "Show the name of companies that have a revenue greater than 100000000000"
response = agent_executor.invoke({"input": user_query})

print("\n--- Agent Response ---")
print(response["output"])

user_query_2 = "Total sales for each month in the year 2024"
response_2 = agent_executor.invoke({"input": user_query_2})
print("\n--- Agent Response ---")
print(response_2["output"])

user_query_3 = "List all customer names associated with the salesperson 'Kosith Theingtrong'."
response_3 = agent_executor.invoke({"input": user_query_3})
print("\n--- Agent Response ---")
print(response_3["output"])

# ***** Additional Examples: Let the Agent use other Tools from the Toolkit *****
# These examples might not result in SQL, but demonstrate the Agent's ability to use other tools.
user_query_4 = "What tables are in the database?"
response_4 = agent_executor.invoke({"input": user_query_4})
print("\n--- Agent Response (List Tables) ---")
print(response_4["output"])

user_query_5 = "Describe the 'contacts' table."
response_5 = agent_executor.invoke({"input": user_query_5})
print("\n--- Agent Response (Describe Table) ---")
print(response_5["output"])

# %%