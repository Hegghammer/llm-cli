# llm-cli

`llm-cli` is a command line utility for interacting with large language model APIs. It was originally developed for use with [Espanso](https://espanso.org) text expander via the [`espanso-llm`](https://gitlab.com/Hegghammer/espanso-llm) extension, but it can also be used on its own.

## Features

- Handles most mainstream LLM API endpoints, both proprietary and local/open-source.
- Can add web search to any language model.
- Can add link following to any language model.
- Provides basic RAG functionality for local document collections.
- Can read prompts from the clipboard.

## Installation 

1. Clone the repository.

```bash
git clone https://github.com/Hegghammer/llm-cli.git
```

2. Enter the `llm-cli` directory and run pip install.

```bash
cd llm-cli
pip install .
```

Run `llm-cli -h` to check that it installed correctly.

## Usage examples

A basic call to an OpenAI model:

```
llm-cli \
--provider-endpoint https://api.openai.com/v1/chat/completions \
--provider-api-key $OPENAI_API_KEY \
--model gpt-4o \
--disallow-clipboard \
--db-file ~/llm-cli.db \
--log-file ~/llm-cli.log \
--prompt "Why is the sky blue"
``` 

A call to an Ollama-served model with web search via Jina: 

```
llm-cli \
--provider-endpoint http://localhost:11434/v1/chat/completions \
--provider-api-key $OLLAMA_API_KEY \
--model gemma3:4b \
--disallow-clipboard \
--db-file ~/llm-cli.db \
--log-file ~/llm-cli.log \
--web-search \
--search-api-key $JINA_API_KEY \
--prompt "Who is the President of Namibia now?"
``` 

A call to Mistral with link following via Jina:

```
llm-cli \
--provider-endpoint https://api.mistral.ai/v1/chat/completions \
--provider-api-key $MISTRAL_API_KEY \
--model mistral-large-latest \
--disallow-clipboard \
--db-file ~/llm-cli.db \
--log-file ~/llm-cli.log \
--follow-links \
--page-reader-api-key $JINA_API_KEY \
--prompt "Summarize this article: https://example.com/article" 
```

RAG on local documents using an Ollama served model:

```
llm-cli \
--rag \
--rag-endpoint http://localhost:11434 \
--rag-embed-model mxbai-embed-large:latest \
--rag-answer-model gemma3:4b \
--rag-doc-dir ~/my_docs \
--rag-db-dir ~/chromadb \
--rag-index-files \ # only necessary on first run
--disallow-clipboard \
--db-file ~/llm-cli.db \
--log-file ~/llm-cli.log \
--prompt "<collection-specific-question>"
```

Call with preset values in `config.ini`:

```
llm-cli \
--config-file config.ini \
--prompt "Why is the sky blue?"
```


## Reference

```
usage: llm-cli [-h] [--allow-clipboard] [--api-key API_KEY]
               [--config-file CONFIG_FILE]
               [--conversation-timeout-minutes CONVERSATION_TIMEOUT_MINUTES]
               [--db-file DB_FILE] [--disallow-clipboard] [--follow-links]
               [--full-content] [--log-file LOG_FILE]
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--model MODEL] [--page-reader-endpoint PAGE_READER_ENDPOINT]
               [--page-reader-api-key PAGE_READER_API_KEY] [--prompt PROMPT]
               [--provider-endpoint PROVIDER_ENDPOINT] [--rag]
               [--rag-answer-model RAG_ANSWER_MODEL] [--rag-db-dir RAG_DB_DIR]
               [--rag-doc-dir RAG_DOC_DIR] [--rag-embed-model RAG_EMBED_MODEL]
               [--rag-endpoint RAG_ENDPOINT]
               [--rag-excerpt-length RAG_EXCERPT_LENGTH] [--rag-index-files]
               [--rag-source-linkformat {wikilinks,markdown}]
               [--search-endpoint SEARCH_ENDPOINT]
               [--search-api-key SEARCH_API_KEY] [--system-role SYSTEM_ROLE]
               [--temperature TEMPERATURE] [--web-search]

Command-line tool to interact with large language models.

options:
  -h, --help            show this help message and exit
  --allow-clipboard     allow clipboard content to be sent to the llm
  --api-key API_KEY     API key to llm provider
  --config-file CONFIG_FILE
                        path to the config file
  --conversation-timeout-minutes CONVERSATION_TIMEOUT_MINUTES
                        conversation timeout in minutes
  --db-file DB_FILE     path to the chat history database file
  --disallow-clipboard  disallow clipboard content to be sent to the llm
  --follow-links        look up the URLs in the prompt
  --full-content        retrieve full webpage content in web search
  --log-file LOG_FILE   path to the log file
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        specify the logging level
  --model MODEL         name of llm
  --page-reader-endpoint PAGE_READER_ENDPOINT
                        API address for page reader provider
  --page-reader-api-key PAGE_READER_API_KEY
                        API key for page reader provider
  --prompt PROMPT       prompt to the llm
  --provider-endpoint PROVIDER_ENDPOINT
                        API address to llm provider
  --rag                 Use RAG
  --rag-answer-model RAG_ANSWER_MODEL
                        name of llm to be used for RAG
  --rag-db-dir RAG_DB_DIR
                        path to persist directory for RAG
  --rag-doc-dir RAG_DOC_DIR
                        path to documents directory for RAG
  --rag-embed-model RAG_EMBED_MODEL
                        embedding model name for RAG.
  --rag-endpoint RAG_ENDPOINT
                        API address for RAG models provider
  --rag-excerpt-length RAG_EXCERPT_LENGTH
                        length (in characters) of excerpts in RAG answer
  --rag-index-files     re-index documents for RAG from scratch
  --rag-source-linkformat {wikilinks,markdown}
                        format of source links in RAG answer
  --search-endpoint SEARCH_ENDPOINT
                        API address for web search provider
  --search-api-key SEARCH_API_KEY
                        API key for web search provider
  --system-role SYSTEM_ROLE
                        system role for llm prompt
  --temperature TEMPERATURE
                        llm temperature
  --web-search          add web search
```

## Acknowledgments

Some the source code is derived from @rohitna's [chatgpt-script](https://github.com/rohitna/chatgpt-script).

