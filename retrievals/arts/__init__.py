from dotenv import load_dotenv

load_dotenv(dotenv_path='./.env')

# Boilerplate commands
from .boilerplate import (
    new_workspace,
    include_general,
    copy_agendas,
    set_basics,
    setup_spectroscopy,
)
