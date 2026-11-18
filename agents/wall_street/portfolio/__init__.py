"""Read-only access to the user's portfolio.

Design rule: this package can READ holdings but has no way to place orders. There is no
order code anywhere in the project, so the agent cannot trade - not by accident, not
because a prompt told it to.
"""
