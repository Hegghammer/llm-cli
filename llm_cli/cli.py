import argparse
import os
from pathlib import Path
from . import core

def main():

    args = core.parse_args()
    config = {}
    config["config_file"] = args.config_file
    config_file = config.get("config_file")
    if config_file:
        config = core.read_config(config_file)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # config["log_file"] = Path(config["log_file"]).expanduser()
    # config["db_file"] = Path(config["db_file"]).expanduser()
    # config["rag_db_dir"] = Path(config["rag_db_dir"]).expanduser()
    # config["rag_doc_dir"] = Path(config["rag_doc_dir"]).expanduser()
    
    core.setup_logging(config["log_file"], config["log_level"])

    core.run_model(config)

if __name__ == "__main__":
    main()
