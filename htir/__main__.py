"""Enable ``python -m htir`` as an alias for the ``htir`` CLI."""

from htir.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
