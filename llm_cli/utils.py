import argparse
import base64
import configparser
import datetime
import glob
import json
import logging
import os
from pathlib import Path
import pprint
import pyperclip
import re
import requests
import sqlite3
import sys
import time
from typing import List, Tuple, Dict
from chromadb.config import Settings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def setup_logging(logpath: Path, log_level: str) -> None:
    logging.basicConfig(
        filename=logpath,
        filemode='a',
        level=logging.getLevelName(log_level.upper()),
        format="%(asctime)s - %(name)s - [%(levelname)s]: %(message)s",
    )

def get_clipboard_content() -> str:
    try:
        return pyperclip.paste()
    except pyperclip.PyperclipException as e:
        logging.error(f"Clipboard error: {e}")
        return ""

def clear_clipboard() -> None:
    try:
        pyperclip.copy("")
    except pyperclip.PyperclipException as e:
        logging.error(f"Unable to clear clipboard: {e}")

def encode_base64(s: str) -> str:
    try:
        return base64.b64encode(s.encode()).decode()
    except Exception as e:
        print(f"Error encoding string to base64: {e}")
        return ""

def decode_base64(s: str) -> str:
    try:
        return base64.b64decode(s.encode()).decode()
    except Exception as e:
        print(f"Error encoding string to base64: {e}")
        return ""

def timestamp_n_minutes_ago(n: int) -> int:
    try:
        return int((datetime.datetime.utcnow() - datetime.timedelta(minutes=n)).timestamp())
    except Exception as e:
        print(f"Error calculating timestamp: {e}")
        return 0

def setup_database(db_file: Path) -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    );"""
    try:
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(schema)
            conn.commit()
        logging.info(f"Database initialized successfully at {db_file}")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")

def insert_conversation(role: str, content: str, db_file: Path) -> None:
    try:
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            timestamp = int(time.time())
            cursor.execute(
                "INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
                (timestamp, encode_base64(role), encode_base64(content)),
            )
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        
def get_conversations_after_timestamp(timestamp: int, db_file: Path) -> Tuple[List[str], List[str]]:
    try:
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM conversations WHERE timestamp >= ?", (timestamp,)
            )
            rows = cursor.fetchall()
            roles = [decode_base64(row[0]) for row in rows]
            contents = [decode_base64(row[1]) for row in rows]
        return roles, contents
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        
def send_request(json_data: str, provider_endpoint: str, provider_api_key: str) -> Dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {provider_api_key}"}
    try:
        response = requests.post(provider_endpoint, data=json_data, headers=headers)
        response.raise_for_status()
        resp = response.json()
        logging.debug(f"Response received from the LLM:\n{pprint.pformat(resp, indent=2)}")
        return resp
    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
        raise
    except requests.exceptions.RequestException as err:
        logging.error(f"An error occurred while sending the request: {err}")
        raise

def make_searchphrase(prompt: str, model: str, provider_endpoint: str, provider_api_key: str, temperature: float, system_role: str) -> str:
    query_prompt = f'''
    You are an intelligent and resourceful assistant for web searching. Your goal is to help users generate good search queries based on their questions. Respond only with the exact text for optimal web searching.

    Examples:
    Q: What is the capital of France?
    A: Capital of France

    Q: How do I use the SearxNG API?
    A: SearxNG API documentation

    Q: Who won the Nobel Peace Prize in 2024?
    A: Nobel Peace Prize laureate 2024

    Here is the user's input:

    {prompt}
    '''
    try:
        messages = [{"role": "system", "content": system_role}, {"role": "user", "content": query_prompt}]
        data = {"model": model, "temperature": temperature, "messages": messages}
        json_data = json.dumps(data)
        response = send_request(json_data, provider_endpoint, provider_api_key)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error contacting LLM: {e}")
        return ""

def web_search(query: str, search_endpoint: str, search_api_key: str, full_content: bool = False) -> Dict:
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {search_api_key}',
        "X-Timeout": "10"
    }
    if full_content:
        headers['X-Engine'] = 'direct'
        logging.info("Using full content web search.")
    else:
        headers['X-Respond-With'] = 'no-content'

    params = {'q': query, 'format': "json"}

    try:
        response = requests.get(search_endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return {}

def get_urls(text: str) -> List[str]:
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    try:
        return re.findall(url_pattern, text)
    except re.error as e:
        print(f"Regex error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def check_for_unreachable(urls: List[str]) -> List[str]:
    unreachable = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                unreachable.append(url)
        except requests.RequestException:
            unreachable.append(url)
    return unreachable

def get_page_text(url: str, page_reader_endpoint: str, page_reader_api_key: str) -> str:
    req_url = f"{page_reader_endpoint.rstrip('/')}/{url}"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {page_reader_api_key}',
        'DNT': '1',
        'X-Engine': 'direct',
        'X-Locale': 'en-GB',
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
        'X-With-Links-Summary': 'all'
    }
    try:
        response = requests.get(req_url, headers=headers)
        response.raise_for_status()
        return response.json()['data']['content']
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except KeyError as key_err:
        print(f"Key error occurred: {key_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return ""

def get_url_contents(urls: List[str], page_reader_endpoint: str, page_reader_api_key: str) -> str:
    results = []
    for url in urls:
        try:
            contents = get_page_text(url, page_reader_endpoint, page_reader_api_key)
            results.append({"url": url, "content": contents})
        except Exception as e:
            print(f"Error fetching content for {url}: {e}")
            results.append({"url": url, "content": ""})
    return json.dumps(results, indent=4)

def load_and_index_docs(rag_doc_dir: Path, rag_db_dir: Path, embeddings) -> Chroma:
    try:
        #files = glob.glob(os.path.join(rag_doc_dir, "*.md")) # nonrecursive
        files = glob.glob(os.path.join(rag_doc_dir, '**', '*.md'), recursive=True)
        docs = []
        for file in files:
            try:
                loaded_docs = UnstructuredMarkdownLoader(file).load()
                filename = os.path.basename(file)
                for doc in loaded_docs:
                    doc.page_content = f"Filename: {filename}\n\n" + doc.page_content
                    doc.metadata["source"] = os.path.basename(file)
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading document {file}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=rag_db_dir)
        return vectorstore
    except Exception as e:
        print(f"Error during indexing documents: {e}")
        raise

def get_centered_excerpt(text: str, answer: str, length: int) -> str:
    try:
        idx = text.lower().find(answer.lower())
        if idx >= 0:
            start = max(idx - length // 2, 0)
            end = start + length
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
            return excerpt
        else:
            return text[:length].strip() + ("..." if len(text) > length else "")
    except Exception as e:
        print(f"Error generating excerpt: {e}")
        return text[:length].strip() + ("..." if len(text) > length else "")

def format_source_link(path: Path, style: str, rag_doc_dir: str) -> str:
    try:
        filename = os.path.splitext(os.path.basename(path))[0]
        if style == "wikilinks":
            return f"[[{filename}]]"
        else:  # markdown
            full_path = f"{rag_doc_dir.rstrip('/')}/{path}"
            return f"[{filename}]({full_path})"
    except Exception as e:
        print(f"Error formatting source link: {e}")
        return f"[{path}]"

def read_config(config_file: Path) -> Dict:
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        return {
            "allow_clipboard": config.getboolean("default", "allow_clipboard", fallback=True),
            "conversation_timeout_minutes": config.getint("default", "conversation_timeout_minutes", fallback=10),
            "db_file": config.get("default", "db_file", fallback=""),
            "log_file": config.get("default", "log_file", fallback=""),
            "log_level": config.get("default", "log_level", fallback="INFO"),
            "model": config.get("default", "model", fallback=""),
            "page_reader_api_key": config.get("default", "page_reader_api_key", fallback=""),
            "page_reader_endpoint": config.get("default", "page_reader_endpoint", fallback="https://eu.r.jina.ai/"),
            "provider_endpoint": config.get("default", "provider_endpoint", fallback=""),
            "provider_api_key": config.get("default", "provider_api_key", fallback=""),
            "rag_answer_model": config.get("rag", "rag_answer_model", fallback=""),
            "rag_db_dir": config.get("rag", "rag_db_dir", fallback=""),
            "rag_doc_dir": config.get("rag", "rag_doc_dir", fallback=""),
            "rag_embed_model": config.get("rag", "rag_embed_model", fallback=""),
            "rag_endpoint": config.get("rag", "rag_endpoint", fallback=""),
            "rag_excerpt_length": config.get("rag", "rag_excerpt_length", fallback="100"),
            "rag_source_linkformat": config.get("rag", "rag_source_linkformat", fallback="markdown"),
            "search_api_key": config.get("default", "search_api_key", fallback=""),
            "search_endpoint": config.get("default", "search_endpoint", fallback="https://eu.s.jina.ai/"),
            "system_role": config.get("default", "system_role", fallback="A helpful assistant"),
            "temperature": config.get("default", "temperature", fallback=1),
        }
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)
