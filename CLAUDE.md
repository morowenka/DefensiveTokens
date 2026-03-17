Solve tasks as quickly and simply as possible. Minimize code changes. Keep code short and clear.
Prioritize modular structure over mega-files.
Solve the task directly, without unnecessary layers. Do not add functionality “for the future.” Avoid overengineering.

Avoid hardcoding except temporal files.
Avoid fallbacks. Skip unnecessary conditions. Do not use default values to handle errors.
Do not add validation for the presence of every field. Trust the data structure.

Follow OOP and SOLID principles:
* **Encapsulation**: Hiding the internal state of an object behind public methods
* **Inheritance**: Code reuse through class hierarchies
* **Polymorphism**: Unified work with objects of different types
* **Abstraction**: Highlighting essential characteristics, hiding details
SOLID principles:
* **S** — Single Responsibility
* **O** — Open/Closed
* **L** — Liskov Substitution
* **I** — Interface Segregation
* **D** — Dependency Inversion

Use raise instead of fallbacks. Allow exceptions to bubble up. Do not return None or default values on errors.
Use try-except to handle expected specific errors only where needed. Do not wrap code into excessive blocks.
Use YAML for configuration files. Before create another config file search repo for existing config files, that can be used.
Use absolute paths instead of relative ones.
Use the standard `logging` module. Save logs to the `logs/` directory. Do not use print for logging.
Use snake_case for files and functions. Use clear names. Avoid abbreviations.

Use `uv` for dependency management. Dependencies are in `pyproject.toml`. Use exact package versions. First install using `uv pip install` without specifying a version, then add it to the project using `uv add` with the installed version.
DO NOT WRITE TESTS. ANY.

You may create temporary files and scripts when testing hypotheses.
Experimental code may be less structured. Hardcoding is allowed for quick idea validation.
After confirming or rejecting a hypothesis, delete temporary files and scripts. Move useful working code into the main project structure if the hypothesis implies it.

Do not use emoji.