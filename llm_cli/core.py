import argparse
from chromadb.config import Settings
import datetime
import json
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
import logging
import os
from pathlib import Path
import pprint
from .utils import (
    setup_logging, read_config, get_clipboard_content, clear_clipboard, setup_database,
    insert_conversation, get_conversations_after_timestamp, send_request, load_and_index_docs,
    make_searchphrase, web_search, get_urls, check_for_unreachable, get_url_contents, 
    get_centered_excerpt, format_source_link, timestamp_n_minutes_ago
)

def parse_args():
    parser = argparse.ArgumentParser(description="Command-line tool to interact with large language models.")
    parser.add_argument("--allow-clipboard", action="store_true", help="allow clipboard content to be sent to the llm", default=True)
    parser.add_argument("--config-file", help="path to the config file")
    parser.add_argument("--conversation-timeout-minutes", default=10, type=int, help="conversation timeout in minutes")
    parser.add_argument("--db-file", help="path to the chat history database file")
    parser.add_argument("--disallow-clipboard", dest="allow_clipboard", action="store_false", help="disallow clipboard content to be sent to the llm")
    parser.add_argument("--follow-links", action="store_true", help="look up the URLs in the prompt")
    parser.add_argument('--full-content', action='store_true', help='retrieve full webpage content in web search')
    parser.add_argument("--log-file", help="path to the log file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="specify the logging level")
    parser.add_argument("--model", help="name of llm")
    parser.add_argument("--page-reader-endpoint", default="https://eu.r.jina.ai/", help="API address for page reader provider")
    parser.add_argument("--page-reader-api-key", help="API key for page reader provider")    
    parser.add_argument("--prompt", default="", help="prompt to the llm")
    parser.add_argument("--provider-api-key", help="API key to llm provider")
    parser.add_argument("--provider-endpoint", help="API address to llm provider")
    parser.add_argument("--rag", action="store_true", help="Use RAG")
    parser.add_argument("--rag-answer-model", help="name of llm to be used for RAG")
    parser.add_argument("--rag-db-dir", help="path to persist directory for RAG")
    parser.add_argument("--rag-doc-dir", help="path to documents directory for RAG")
    parser.add_argument("--rag-embed-model", help="embedding model name for RAG.")
    parser.add_argument("--rag-endpoint", help="API address for RAG models provider")
    parser.add_argument("--rag-excerpt-length", default=200, type=int, help="length (in characters) of excerpts in RAG answer")
    parser.add_argument("--rag-index-files", action="store_true", help="re-index documents for RAG from scratch")
    parser.add_argument("--rag-source-linkformat", default="markdown", choices=["wikilinks", "markdown"], help="format of source links in RAG answer")
    parser.add_argument("--search-endpoint", default="https://eu.s.jina.ai/", help="API address for web search provider")
    parser.add_argument("--search-api-key", help="API key for web search provider")
    parser.add_argument("--system-role", default="A helpful assistant", help="system role for llm prompt")
    parser.add_argument("--temperature", default=1, type=float, help="llm temperature")
    parser.add_argument("--web-search", action="store_true", help="add web search")
    return parser.parse_args()

def run_model(config: dict):
    setup_database(config["db_file"])
    if not config["allow_clipboard"]:
        clear_clipboard()
    clipboard_content = get_clipboard_content()

    if not clipboard_content and not config["prompt"]:
        raise ValueError("The new prompt is empty")
    elif not clipboard_content:
        new_prompt = config["prompt"]
    elif not config["prompt"]:
        new_prompt = clipboard_content
    else:
        new_prompt = f"{config['prompt']}: {clipboard_content}"

    if config["rag"]:
        logging.info(f'Entering RAG mode on the document collection in {config["rag_doc_dir"]}')
        embeddings = OllamaEmbeddings(base_url=config["rag_endpoint"], model=config["rag_embed_model"])
        llm = ChatOllama(base_url=config["rag_endpoint"], model=config["rag_answer_model"], temperature=config["temperature"])

        if config["rag_index_files"] or not os.path.exists(config["rag_db_dir"]):
            logging.debug(f'Indexing documents in {config["rag_db_dir"]}')
            vectorstore = load_and_index_docs(config["rag_doc_dir"], config["rag_db_dir"], embeddings)
        else:
            db = f"{config['rag_db_dir']}/chroma.sqlite3"
            last_modtime = datetime.datetime.fromtimestamp(os.path.getmtime(db)).strftime("%e %b %Y at %H:%M")
            logging.debug(f'Loading existing vectorstore from {config["rag_db_dir"]} (last indexed {last_modtime})')
            vectorstore = Chroma(persist_directory=config["rag_db_dir"], embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False))

        template = """
        Use the provided context to answer the question concisely.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": new_prompt})
        logging.debug(f"Full response received from the LLM:\n{pprint.pformat(response, indent=2)}")

        reply_dict = {
            "answer": response['result'],
            "sources": []
        }

        seen = set()
        for doc in response["source_documents"]:
            full_text = doc.page_content.replace('\n', ' ').strip()
            source_file = doc.metadata["source"]
            excerpt = get_centered_excerpt(full_text, response['result'], config["rag_excerpt_length"])
            link = format_source_link(source_file, config["rag_source_linkformat"], config["rag_doc_dir"])
            key = (excerpt, link)
            if key not in seen:
                seen.add(key)
                reply_dict["sources"].append((excerpt, link))

        reply = f"\nAnswer: {reply_dict['answer']}\n\nSources:\n"
        for excerpt, link in reply_dict["sources"]:
            reply += f'- "{excerpt}" {link}\n'
        logging.info(f"Answer received from the LLM:\n{reply}")
        print(reply)

    else:
        if config["web_search"]:
            logging.info(f"Searching the web ..")
            today = datetime.datetime.today().strftime("%d %B %Y")
            search_phrase = make_searchphrase(new_prompt, config["model"], config["provider_endpoint"], config["provider_api_key"], config["temperature"], config["system_role"])
            logging.info(f'''
                        Entering web search mode with the following parameters:
                        - search_endpoint: {config["search_endpoint"]}
                        - search_phrase: {search_phrase}
                        ''')
            search_results = web_search(search_phrase, config["search_endpoint"], config["search_api_key"], full_content=config["full_content"])
            new_prompt = f'''
                You are an intelligent and resourceful assistant. Your goal is to answer the user's question using the results from a web search.
                Here is the user's original question: {new_prompt}
                Today's date is {today}.
                Here is the information retrieved from the web search, in json format. The text of the page is in the "description" or "content" field.
                ------------
                {search_results}
                ------------
                Now use this context to answer the user's original question in a clear manner. Be very concise. At the end, list the titles, urls, and publication dates of the three main sources you relied on to answer the question. Use markdown format like so: * [title](url) (dd mmm YYYY).
                '''

        if config["follow_links"]:
            urls = get_urls(new_prompt)
            logging.info(f'Entering link following mode for the following URLs:\n{urls}')
            unreachable = check_for_unreachable(urls)
            if unreachable:
                logging.error("Some of the links could not be reached. Check the following urls and try again.")
                for u in unreachable:
                    logging.error(f"- {u}")
                sys.exit(0)
            else:
                url_contents = get_url_contents(urls, config["page_reader_endpoint"], config["page_reader_api_key"])
                new_prompt = f'''
                    You are an intelligent and resourceful assistant. Your goal is to answer the user's question using the contents of the webpages mentioned in his question.
                    Here is the user's original question:
                    ----
                    {new_prompt}
                    ----
                    Here is the content of the webpages mentioned in the question, in json format:
                    ------------
                    {url_contents}
                    ------------
                    Now use this context to answer the user's original question in a clear manner. Be very concise.
                    '''

        timestamp_cutoff = timestamp_n_minutes_ago(config["conversation_timeout_minutes"])
        messages = (
            [{"role": "system", "content": config["system_role"]}]
            + [
                {"role": role, "content": content}
                for role, content in zip(
                    *get_conversations_after_timestamp(timestamp_cutoff, config["db_file"])
                )
            ]
            + [{"role": "user", "content": new_prompt}]
        )

        logging.debug(f"Messages to be sent to the LLM:\n{pprint.pformat(messages, indent=2)}")

        data = {"model": config["model"], "temperature": config["temperature"], "messages": messages}
        json_data = json.dumps(data)

        response = send_request(json_data, config["provider_endpoint"], config["provider_api_key"])
        reply = response["choices"][0]["message"]["content"]
        logging.info(f"Answer received from the LLM:\n{reply}")
        print(reply)

    insert_conversation("user", new_prompt, config["db_file"])
    insert_conversation("assistant", reply, config["db_file"])

