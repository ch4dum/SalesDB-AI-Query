#%% Connect with PostgreSQL
from langchain_community.utilities import SQLDatabase
import torch
from langchain_community.llms import Ollama
import psycopg2
#%%
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
WHERE table_schema = 'd365' AND table_type = 'BASE TABLE';
""")

tables = [f'd365.{row[0]}' for row in cursor.fetchall()]
print("Tables in d365:", tables)
#%%
# Connect with PostgreSQL
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

engine = create_engine(
    "postgresql://postgres:Csi%40dmin9@localhost:5432/d365_database",
    connect_args={"options": "-c search_path=d365"}
)

db = SQLDatabase(engine=engine)

# Cuda check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

llm = Ollama(model="qwen3:4b")

# Test model
response = llm("สวัสดีครับ")
print(response)
# %%
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain

prompt_template = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template="""
You are a PostgreSQL SQL generator. Use only the following schema:

{table_info}

Given the user's question: {input}

Only output valid SQL. Do not explain. Use dialect: {dialect}
"""
)

db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    prompt=prompt_template,
    input_key="input",
    verbose=True
)

table_info = db.get_table_info()

response = db_chain.invoke({
    "input": "อยากได้ยอดขายในปี 2023",
    "table_info": table_info,
    "dialect": "postgresql"
})

print(response["result"])
# %%
print(db.get_usable_table_names())
# %%