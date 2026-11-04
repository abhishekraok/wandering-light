"""Signal House: a local workshop that teaches the `mark-28` stack machine.

The workshop *observes* the canonical machine -- every op is applied through
``wandering_light.stack_machine.step_raw`` -- and never reimplements it. The
core (``engine``, ``curriculum``, ``server``) needs only the standard library;
``model`` wraps an optional ``stack_policy`` checkpoint as an adviser.

A browser frontend is a later change; this package is the Python service.
"""
