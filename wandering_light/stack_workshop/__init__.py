"""Signal House: a local workshop that teaches the `mark-28` stack machine.

``engine`` compiles and runs a learner's program by calling the canonical
machine one op at a time, ``curriculum`` holds the authored lessons, ``model``
is the optional policy adviser, and ``server`` puts them behind a localhost
JSON API. Only ``model`` needs anything beyond the standard library, and only
once a checkpoint is loaded.
"""
