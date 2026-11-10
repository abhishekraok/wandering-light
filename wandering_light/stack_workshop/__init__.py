"""Signal House: a local workshop that teaches the mark-28 stack machine.

The workshop observes the canonical machine -- every step goes through
``stack_machine.step_raw`` -- and never reimplements it. ``engine`` compiles
and runs programs, ``curriculum`` holds the authored lessons, ``model`` is the
optional policy adviser, and ``server`` exposes all of it as a localhost JSON
service for the browser frontend that comes later.

Run with ``uv run python -m wandering_light.stack_workshop.server``.
"""
