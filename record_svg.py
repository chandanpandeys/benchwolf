import sys
import os
import platform
import psutil

# Add src to path just in case
sys.path.insert(0, os.path.abspath("src"))

from inferbench.reporting.console import console
console.record = True
console.width = 95

from inferbench.cli import main

sys.argv = ["inferbench", "preflight"]

print("Generating demo.svg...")
try:
    main()
except SystemExit:
    pass

console.save_svg("demo.svg", title="inferbench preflight")
print("Finished saving demo.svg!")
