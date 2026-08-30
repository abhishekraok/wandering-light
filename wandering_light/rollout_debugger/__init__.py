"""An interactive debugger for the materialized-state rollout solver.

The search it shows is the production one: ``trace`` records what
``RolloutPredictor.search_with_cost`` decides through an observer that cannot
steer it, and the encoding panel calls the policy's own ``pair_batch``. Nothing
here reimplements search or encoding, so what the debugger displays is what the
CLI solver did.
"""
