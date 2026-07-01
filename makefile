# Define the interpreter
PYTHON = python

# Define the source files
SRC2 = main_section2.py
SRC3 = main_section3.py
SRC3c = main_section3c.py
SRC3d = main_section3d.py
SRC5 = main_section5.py

# Define the virtual environment directory
VENV_DIR = .venv

# Check and activate virtual environment if necessary
ACTIVATE_VENV = \
    if [ -z "$$VIRTUAL_ENV" ]; then \
        . $(VENV_DIR)/bin/activate; \
    fi

# Start the compilation (execution in this case)
all: section2 section3 section3c section3d section5

section2:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC2)

section3:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC3)

section3c:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC3c)

section3d:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC3d)

section5:
	$(ACTIVATE_VENV) && $(PYTHON) $(SRC5)
