import os, sys
# Ensure this custom_nodes package root is on sys.path so internal helper packages
# like "dual_ksampler" can be imported as top-level modules.
sys.path.insert(0, os.path.dirname(__file__))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
