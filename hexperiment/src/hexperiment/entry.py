"""The ``hexperiment`` console script: a light door in front of ``cli``.

``cli`` imports ``canary`` and so pandas and NumPy. The stop check runs at the end
of every agent turn and needs none of them, so it is answered here without
importing ``cli``. The same holds for ``guard-write``, which runs before every
file write of the ghost hunter. Every other argument list goes to ``cli.main``
unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``hexperiment``; ``hook --stop`` takes the light path."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["hook", "--stop"]:
        from hexperiment.hook import stop_command

        return stop_command()
    if args == ["guard-write"]:
        from hexperiment.write_guard import main as guard_main

        return guard_main()
    from hexperiment.cli import main as cli_main

    return cli_main(args)
