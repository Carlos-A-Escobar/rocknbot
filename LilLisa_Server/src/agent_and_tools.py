"""
ReAct agent that handles a query in an intelligent manner
"""

import os
import re
import traceback
from difflib import get_close_matches
from enum import Enum

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

import lancedb

from src import utils
from src.llama_index_lancedb_vector_store import LanceDBVectorStore

OPENAI_API_KEY = None
IDDM_RETRIEVER = None
IDA_RETRIEVER = None
IDDM_QA_PAIRS_RETRIEVER = None
IDA_QA_PAIRS_RETRIEVER = None
RERANKER = SentenceTransformerRerank(top_n=50, model="cross-encoder/ms-marco-MiniLM-L-12-v2")
CLIENT = None
OPENAI_CLIENT = None

QA_SYSTEM_PROMPT = None
QA_USER_PROMPT = None

lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT
if fp := lillisa_server_env["SPEEDICT_FOLDERPATH"]:
    speedict_folderpath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("SPEEDICT_FOLDERPATH not found in lillisa_server.env")
    raise ValueError("SPEEDICT_FOLDERPATH not found in lillisa_server.env")


if fp := lillisa_server_env["OPENAI_API_KEY_FILEPATH"]:
    openai_api_key_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("OPENAI_API_KEY_FILEPATH not found in lillisa_server.env")
    raise ValueError("OPENAI_API_KEY_FILEPATH not found in lillisa_server.env")

with open(openai_api_key_filepath, "r", encoding="utf-8") as file:
    OPENAI_API_KEY = file.read()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


if fp := lillisa_server_env["QA_SYSTEM_PROMPT_FILEPATH"]:
    qa_system_prompt_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("QA_SYSTEM_PROMPT_FILEPATH not found in lillisa_server.env")
    raise ValueError("QA_SYSTEM_PROMPT_FILEPATH not found in lillisa_server.env")

with open(qa_system_prompt_filepath, "r", encoding="utf-8") as file:
    QA_SYSTEM_PROMPT = file.read()


if fp := lillisa_server_env["QA_USER_PROMPT_FILEPATH"]:
    qa_user_prompt_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("QA_USER_PROMPT_FILEPATH not found in lillisa_server.env")
    raise ValueError("QA_USER_PROMPT_FILEPATH not found in lillisa_server.env")

with open(qa_user_prompt_filepath, "r", encoding="utf-8") as file:
    QA_USER_PROMPT = file.read()


if fp := lillisa_server_env["AWS_ACCESS_KEY_ID_FILEPATH"]:
    aws_access_key_id_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("AWS_ACCESS_KEY_ID not found in lillisa_server.env")
    raise ValueError("AWS_ACCESS_KEY_ID not found in lillisa_server.env")

with open(aws_access_key_id_filepath, "r", encoding="utf-8") as file:
    aws_access_key_id = file.read()


if fp := lillisa_server_env["AWS_SECRET_ACCESS_KEY_FILEPATH"]:
    aws_secret_access_key_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("AWS_SECRET_ACCESS_KEY not found in lillisa_server.env")
    raise ValueError("AWS_SECRET_ACCESS_KEY not found in lillisa_server.env")


with open(aws_secret_access_key_filepath, "r", encoding="utf-8") as file:
    aws_secret_access_key = file.read()

if not os.path.exists(speedict_folderpath):
    os.makedirs(speedict_folderpath)

if fp := lillisa_server_env["LANCEDB_FOLDERPATH"]:
    lancedb_folderpath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("LANCEDB_FOLDERPATH not found in lillisa_server.env")
    raise ValueError("LANCEDB_FOLDERPATH not found in lillisa_server.env")

if not os.path.exists(lancedb_folderpath):
    traceback.print_exc()
    utils.logger.critical("%s not found", lancedb_folderpath)
    raise NotImplementedError("%s not found" % lancedb_folderpath)  # pylint: disable=consider-using-f-string


# Establish connection to LanceDB
db = lancedb.connect(lancedb_folderpath)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
iddm_table = db.open_table("IDDM")
ida_table = db.open_table("IDA")
iddm_qa_pairs_table = db.open_table("IDDM_QA_PAIRS")
ida_qa_pairs_table = db.open_table("IDA_QA_PAIRS")
iddm_vector_store = LanceDBVectorStore.from_table(iddm_table)
ida_vector_store = LanceDBVectorStore.from_table(ida_table)
iddm_qa_pairs_vector_store = LanceDBVectorStore.from_table(iddm_qa_pairs_table, "vector")
ida_qa_pairs_vector_store = LanceDBVectorStore.from_table(ida_qa_pairs_table, "vector")
IDDM_INDEX = VectorStoreIndex.from_vector_store(vector_store=iddm_vector_store)
IDA_INDEX = VectorStoreIndex.from_vector_store(vector_store=ida_vector_store)
IDDM_QA_PAIRS_INDEX = VectorStoreIndex.from_vector_store(vector_store=iddm_qa_pairs_vector_store)
IDA_QA_PAIRS_INDEX = VectorStoreIndex.from_vector_store(vector_store=ida_qa_pairs_vector_store)


IDDM_RETRIEVER = IDDM_INDEX.as_retriever(similarity_top_k=50)
IDA_RETRIEVER = IDA_INDEX.as_retriever(similarity_top_k=50)
IDDM_QA_PAIRS_RETRIEVER = IDDM_QA_PAIRS_INDEX.as_retriever(similarity_top_k=8)
IDA_QA_PAIRS_RETRIEVER = IDA_QA_PAIRS_INDEX.as_retriever(similarity_top_k=8)


class PRODUCT(str, Enum):
    """Product"""

    IDA = "IDA"
    IDDM = "IDDM"

    @staticmethod
    def get_product(product: str) -> "PRODUCT":
        """get product"""
        if product in (product.value for product in PRODUCT):
            return PRODUCT(product)
        raise ValueError(f"{product} does not exist")


def update_retriever(retriever_name, new_retriever):
    """
    Updates the reference to the appropriate retriever after "rebuild_docs" or "update_golden_qa_pairs" is called.
    """
    global IDDM_RETRIEVER, IDA_RETRIEVER, IDDM_QA_PAIRS_RETRIEVER, IDA_QA_PAIRS_RETRIEVER
    if retriever_name == "IDDM":
        IDDM_RETRIEVER = new_retriever
    elif retriever_name == "IDA":
        IDA_RETRIEVER = new_retriever
    elif retriever_name == "IDDM_QA_PAIRS":
        IDDM_QA_PAIRS_RETRIEVER = new_retriever
    elif retriever_name == "IDA_QA_PAIRS":
        IDA_QA_PAIRS_RETRIEVER = new_retriever
    else:
        raise ValueError(f"{retriever_name} does not exist")


def handle_user_answer(answer: str) -> str:
    """
    Tool should be caleld when a user enters an answer to a previous question of theirs. Thank them and merely mimic their answer.
    """
    return answer


def improve_query(query: str, conversation_history: str) -> str:
    """
    Clears up vagueness from query with the help of the conversation history and returns a new query revealing the user's true intention, without distorting the meaning behind the original query. If needed, this should be the first tool called; else, should not be called at all.
    * query should be the original query that the user prompted the agent with, needing some clarification
    * conversation_history is the conversation history the user prompted the agent with
    """
    user_prompt = f"""
    ###CONVERSATION HISTORY###
    {conversation_history}

    ###QUERY###
    {query}

    Based on the conversation history and query, generate a new query that links the two, maximizing semantic understanding.
    """
    response = (
        OPENAI_CLIENT.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": user_prompt}])
        .choices[0]
        .message.content
    )

    return response


def answer_from_document_retrieval(
    product: str, original_query: str, generated_query: str, conversation_history: str
) -> str:
    """
    RAG Search. Searches through a database of 10,000 documents, and based on a query, returns the top-10 relevant documents and synthesizes an answer.
    * original_query should be the query the user prompted the agent with
    * generated_query should be the query generated from the 'improve_query_tool' if it was called before this tool, else parameter should be passed in as an empty string
    * conversation_history is the conversation history the user prompted the agent with
    """
    response = ""
    qa_system_prompt = QA_SYSTEM_PROMPT
    query = generated_query or original_query

    product_enum = PRODUCT.get_product(product)
    if product_enum == PRODUCT.IDDM:
        product_versions = ["v7.4", "v8.0", "v8.1"]
        version_pattern = re.compile(r"v?\d+\.\d+", re.IGNORECASE)
        document_index = IDDM_INDEX
        qa_pairs_index = IDDM_QA_PAIRS_INDEX
        default_document_retriever = IDDM_RETRIEVER
        default_qa_pairs_retriever = IDDM_QA_PAIRS_RETRIEVER
    else:
        product_versions = ["iap-2.0", "iap-2.2", "iap-3.0", "descartes", "descartes-dev", "version-1.5", "version-16"]
        version_pattern = re.compile(r"\b(?:IAP[- ]\d+\.\d+|version[- ]\d+\.\d+|descartes(?:-dev)?)\b", re.IGNORECASE)
        document_index = IDA_INDEX
        qa_pairs_index = IDA_QA_PAIRS_INDEX
        default_document_retriever = IDA_RETRIEVER
        default_qa_pairs_retriever = IDA_QA_PAIRS_RETRIEVER

    matched_versions = get_matching_versions(query, product_versions, version_pattern)

    if matched_versions:
        qa_system_prompt += f"\n10. Mention the product version(s) you used to craft your response were '{' and '.join(matched_versions)}'"
        lance_filter_documents = " OR ".join(f"(metadata.version = '{version}')" for version in matched_versions)
        lance_filter_qa_pairs = "(metadata.version = 'none') OR " + lance_filter_documents
        document_retriever = document_index.as_retriever(
            vector_store_kwargs={"where": lance_filter_documents}, similarity_top_k=50
        )
        qa_pairs_retriever = qa_pairs_index.as_retriever(
            vector_store_kwargs={"where": lance_filter_qa_pairs}, similarity_top_k=8
        )
    else:
        qa_system_prompt += "\n10. At the beginning of your response, mention that because a specific product version was not specified, information from all available versions was used."
        document_retriever = default_document_retriever
        qa_pairs_retriever = default_qa_pairs_retriever

    qa_nodes = qa_pairs_retriever.retrieve(query)
    exact_qa_nodes = []
    relevant_qa_nodes = []
    potentially_relevant_qa_nodes = []

    # Categorize the nodes
    for node in qa_nodes:
        if 0.9 <= node.score <= 1.0:
            exact_qa_nodes.append(node)
        elif 0.8 <= node.score < 0.9:
            relevant_qa_nodes.append(node)
        elif 0.7 <= node.score < 0.8:
            potentially_relevant_qa_nodes.append(node)

    if exact_qa_nodes:
        response += "Found matching QA pairs that have been verified by an expert!\n"
        for idx, node in enumerate(exact_qa_nodes, start=1):
            response += f"\nMatch {idx}:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}\n"
        return response
    if relevant_qa_nodes:
        response += "Here are some relevant QA pairs that have been verified by an expert!\n"
        for idx, node in enumerate(relevant_qa_nodes, start=1):
            response += f"\nMatch {idx}:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}\n"
        response += "\n\nAfter searching through the documentation database, this was found:\n"
    nodes = document_retriever.retrieve(query)
    reranked_nodes = RERANKER.postprocess_nodes(nodes=nodes, query_str=query)[:10]
    useful_links = list(dict.fromkeys([node.metadata['github_url'] for node in reranked_nodes]))[:3]
    chunks = "\n\n".join(f"{node.text}" for node in reranked_nodes)

    user_prompt = QA_USER_PROMPT.replace("<CONTEXT>", chunks)
    user_prompt = user_prompt.replace("<CONVERSATION_HISTORY>", conversation_history)
    user_prompt = user_prompt.replace("<QUESTION>", original_query)

    response += (
        OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": qa_system_prompt}, {"role": "user", "content": user_prompt}],
        )
        .choices[0]
        .message.content
    )

    response += "\n\nHere are some potentially helpful documentation links:\n"
    response += "\n".join(f"- {link}" for link in useful_links)

    if potentially_relevant_qa_nodes and not relevant_qa_nodes:
        response += "\n\n\n"
        response += "In addition, here are some potentially relevant QA pairs that have been verified by an expert!\n"
        for idx, node in enumerate(potentially_relevant_qa_nodes, start=1):
            response += f"\nMatch {idx}:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}\n"
    return response


def get_matching_versions(query, product_versions, version_pattern):
    """
    Not a tool for the ReAct agent but instead a helper function for "answer_from_document_retrieval.
    Helps extract a version from a query.
    """
    extracted_versions = version_pattern.findall(query)
    matched_versions = []
    for extracted_version in extracted_versions:
        closest_match = get_close_matches(extracted_version, product_versions, n=1, cutoff=0.4)
        if closest_match:
            matched_versions.append(closest_match[0])
    return matched_versions