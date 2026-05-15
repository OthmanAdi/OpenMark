import sys, os
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

# Pre-load env state
print("PRE: NEO4J_DATABASE in os.environ?", "NEO4J_DATABASE" in os.environ,
      "value:", os.environ.get("NEO4J_DATABASE"))

from openmark import config
print("POST: config.NEO4J_DATABASE =", repr(config.NEO4J_DATABASE))
print("POST: os.environ['NEO4J_DATABASE'] =", repr(os.environ.get("NEO4J_DATABASE")))
print("POST: config.NEO4J_URI =", repr(config.NEO4J_URI))
print(".env path used by config:")
import openmark.config as c
print(" ", c.__file__)
