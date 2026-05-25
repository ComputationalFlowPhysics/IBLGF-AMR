from __future__ import annotations
import sys
from iblgf import poisson

result = poisson.run(sys.argv[1])
print(result.measured_linf_error)
print(result.difference)
