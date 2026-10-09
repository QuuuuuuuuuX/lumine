"""The built-in 3D open-world sandbox.

Genshin Impact cannot run on this machine, so the project ships its own world. Importing this
package is the only thing that requires the rasteriser; ``lumine.agent`` and everything under
``lumine.brain`` work without it, which is why the top-level package exposes ``SimEnv`` lazily.
"""

from .env import TASK_LIST, TASKS, SimEnv, task_by_id

__all__ = ["SimEnv", "TASKS", "TASK_LIST", "task_by_id"]
