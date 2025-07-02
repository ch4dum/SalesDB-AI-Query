#%% Connect with PostgreSQL
from langchain_community.utilities import SQLDatabase
# import torch
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
# print("Tables in d365:", tables)

cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'd365' AND table_name = 'salesorders';
""")
table_exists = cursor.fetchone()

if table_exists:
    print("Table 'salesorders' exists in schema 'd365'.")
else:
    print("Table 'salesorders' does not exist in schema 'd365'.")

#%%
# Connect with PostgreSQL
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

selected_tables = [
    "contacts",
    "leads",
    "accounts",
    "opportunities",
    "opportunityproducts",
    "quotes",
    "quotedetails",
    "salesorders",
    "salesorderdetails",
    "products",
    "pricelevels",
    "productpricelevels",
    "appointments",
    "emails",
    "phonecalls"
]

engine = create_engine(
    "postgresql://postgres:Csi%40dmin9@localhost:5432/d365_database",
    connect_args={"options": "-c search_path=d365"}
)

db = SQLDatabase(engine=engine, include_tables=selected_tables)

print(db.get_usable_table_names())

# # Cuda check
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Running on device: {device}")

# llm = Ollama(model="qwen3:4b")

# # Test model
# response = llm("สวัสดีครับ")
# print(response)
# %%
# Query SQL
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")
model = AutoModelForCausalLM.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")

# %%

prompt_template = PromptTemplate(
    input_variables=["input", "dialect", "table_info"],
    template="""
You are a helpful AI that generates only valid PostgreSQL queries.
Do not include any explanation or placeholders like <think>, just output pure SQL.
You are a PostgreSQL SQL generator. Use only the schema name "d365".

Here is information about the tables you can query:
{table_info}

Given the user's question: {input}

Only output valid SQL. Do not explain. Use dialect: {dialect}
"""
)

table_info_string = """
Table Name: contacts

Description: This table stores contact information for customers and leads, including personal details, communication methods, and organizational links for sales management.

Top 5 Columns in the Table:
1.  fullname
2.  emailaddress1
3.  mobilephone
4.  jobtitle
5.  _parentcustomerid_value

---

Table Name: leads

Description: This table stores information about potential customers or prospects, capturing details relevant to initial sales engagement and qualification. It includes contact information, lead source, budget, and sales stage to track progress.

Top 5 Columns in the Table:
1.  fullname
2.  emailaddress1
3.  mobilephone
4.  companyname
5.  leadqualitycode

---

Table Name: accounts

Description: This table stores detailed information about companies, acting as the central hub for organizational data. It includes financial figures, industry classifications, addresses, and key contacts, essential for comprehensive customer relationship management.

Top 5 Columns in the Table:
1.  name
2.  telephone1
3.  websiteurl
4.  revenue
5.  _primarycontactid_value

---

Table Name: opportunities

Description: This table tracks potential sales deals, detailing various aspects from initial stages to closure. It includes estimated values, sales stages, customer needs, and financial information relevant to each opportunity.

Top 5 Columns in the Table:
1.  name
2.  estimatedvalue
3.  estimatedclosedate
4.  salesstage
5.  _customerid_value

---

Table Name: opportunityproducts

Description: This table details the specific products or services included within a sales opportunity. It captures information like product name, quantity, pricing, and associated discounts, providing a breakdown of the opportunity's total value.

Top 5 Columns in the Table:
1.  productname
2.  quantity
3.  priceperunit
4.  extendedamount
5.  _opportunityid_value

---

Table Name: quotes

Description: This table stores sales quotes generated for customers, detailing proposed products/services, pricing, discounts, and tax information. It also tracks the quote's status and links to related opportunities and accounts.

Top 5 Columns in the Table:
1.  name
2.  quotenumber
3.  totalamount
4.  statuscode
5.  _customerid_value

---

Table Name: quotedetails

Description: This table stores line-item information for each product or service on a sales quote. It includes specific details such as quantity, pricing, discounts, and the product itself, contributing to the overall quote value.

Top 5 Columns in the Table:
1.  productname
2.  quantity
3.  priceperunit
4.  extendedamount
5.  _quoteid_value

---

Table Name: salesorders

Description: This table contains details for sales orders, which are confirmed customer purchases. It includes financial sums like total amount, discounts, and taxes, along with links to the original quote, opportunity, and customer.

Top 5 Columns in the Table:
1.  name
2.  ordernumber
3.  totalamount
4.  statuscode
5.  _customerid_value

---

Table Name: salesorderdetails

Description: This table provides line-item specifics for products or services included in a sales order. It covers details like **product name**, **quantity**, **price per unit**, and **extended amounts**, giving a granular view of each ordered item.

Top 5 Columns in the Table:
1.  productname
2.  quantity
3.  priceperunit
4.  extendedamount
5.  _salesorderid_value

---

Table Name: products

Description: This table contains a catalog of products and services offered, including their names, unique identifiers, pricing details, and validity dates. It serves as the master list for items that can be included in quotes and sales orders.

Top 5 Columns in the Table:
1.  name
2.  productnumber
3.  standardcost
4.  validfromdate
5.  validtodate

---

Table Name: pricelevels

Description: This table defines different pricing tiers or lists for products and services. It includes the name of the price list, its validity period (begin and end dates), and the associated currency, allowing for varied pricing strategies.

Top 5 Columns in the Table:
1.  name
2.  pricelevelid
3.  begindate
4.  enddate
5.  _transactioncurrencyid_value

---

Table Name: productpricelevels

Description: This table defines the specific pricing for individual products within a given price level. It links products to price lists, specifying the exact amount for each product, the unit of measure, and the currency.

Top 5 Columns in the Table:
1.  _productid_value
2.  _pricelevelid_value
3.  amount
4.  _uomid_value
5.  productnumber

---

Table Name: appointments

Description: This table stores details about scheduled meetings and activities. It includes information such as the meeting subject, scheduled times, participants, and associated records like accounts or opportunities, providing a comprehensive view of planned interactions.

Top 5 Columns in the Table:
1.  subject
2.  scheduledstart
3.  scheduledend
4.  location
5.  _regardingobjectid_value

---

Table Name: emails

Description: This table stores records of email communications, including subject, sender, recipients, and content. It tracks email status and links to related business records like accounts, contacts, or opportunities, providing a log of digital interactions.

Top 5 Columns in the Table:
1.  subject
2.  description
3.  torecipients
4.  senton
5.  _regardingobjectid_value

---

Table Name: phonecalls

Description: This table records details of phone calls made or received. It includes information about the call's subject, duration, scheduled and actual times, and links to related records like contacts or opportunities, providing a log of verbal interactions.

Top 5 Columns:
1.  subject
2.  phonenumber
3.  actualstart
4.  actualdurationminutes
5.  _regardingobjectid_value
"""


def generate_sql(query: str, dialect: str = "postgresql"):
    prompt = prompt_template.format(input=query, dialect=dialect, table_info=table_info_string)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1500)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

user_query = "How many column in salesorders"

# แปลงคำถามเป็น SQL Query
sql = generate_sql(user_query)
print("Generated SQL:", sql)

# %%
# print(db.get_usable_table_names())
# %%