import argparse
import os
from pathlib import Path
from . import core

def main():

    temp = core.parse_args()
    del temp.allow_clipboard, temp.conversation_timeout_minutes, temp.follow_links, temp.full_content, temp.log_level, temp.page_reader_endpoint, temp.prompt, temp.rag, temp.rag_excerpt_length, temp.rag_index_files, temp.rag_source_linkformat, temp.search_endpoint, temp.system_role, temp.temperature, temp.web_search
    if all(value is None for value in vars(temp).values()):
        print("No arguments detected. Run `llm-cli -h` for help.")
        return

    args = core.parse_args()
    config = {}
    config["config_file"] = args.config_file
    config_file = config.get("config_file")
    if config_file:
        config = core.read_config(config_file)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    if "log_file" not in config:
        print("Error: No log filepath specified. You must supply it either in config.ini or\nexplicitly with --log-file. Run `llm-cli -h` for details.")
        return

    if "db_file" not in config:
        print("Error: No database filepath specified. You must supply it either in config.ini or\nexplicitly with --db-file. Run `llm-cli -h` for details.")
        return

    if args.rag is False and "model" not in config:
        print("Error: No model name specified. You must supply it either in config.ini or\nexplicitly with --model. Run `llm-cli -h` for details.")
        return    

    if args.rag is False and "provider_endpoint" not in config:
        print("Error: No provider endpoint specified. You must supply it either in config.ini or\nexplicitly with --provider-endpoint. Run `llm-cli -h` for details.")
        return        

    if args.rag is False and "provider_api_key" not in config:
        print("Error: No api key specified. You must supply it either in config.ini or\nexplicitly with --provider-api-key. Run `llm-cli -h` for details.")
        return  

    if args.rag is True and "rag_doc_dir" not in config:
        print("Error: No documents directory specified. You must supply it either in config.ini or\nexplicitly with --rag-doc-dir. Run `llm-cli -h` for details.")
        return  

    if args.rag is True and "rag_endpoint" not in config:
        print("Error: No RAG endpoint specified. You must supply it either in config.ini or\nexplicitly with --rag-endpoint. Run `llm-cli -h` for details.")
        return  

    if args.rag is True and "rag_embed_model" not in config:
        print("Error: No RAG embedding model specified. You must supply it either in config.ini or\nexplicitly with --rag-embed-model. Run `llm-cli -h` for details.")
        return  

    if args.rag is True and "rag_answer_model" not in config:
        print("Error: No RAG answer model specified. You must supply it either in config.ini or\nexplicitly with --rag-answer-model. Run `llm-cli -h` for details.")
        return  

    if args.rag is True and "rag_db_dir" not in config:
        print("Error: No RAG vector database directory specified. You must supply it either in config.ini or\nexplicitly with --rag-db-dir. Run `llm-cli -h` for details.")
        return  

    if args.web_search is True and "search_api_key" not in config:
        print("Error: No search endpoint API key specified. You must supply it either in config.ini or\nexplicitly with --search-api-key. Run `llm-cli -h` for details.")
        return  

    if args.follow_links is True and "page-reader-api-key" not in config:
        print("Error: No page reader endpoint API key specified. You must supply it either in config.ini or\nexplicitly with --page-reader-api-key. Run `llm-cli -h` for details.")
        return  
    
    core.setup_logging(config["log_file"], config["log_level"])

    core.run_model(config)

if __name__ == "__main__":
    main()
