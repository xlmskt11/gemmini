import sys
from profile_graph import *

if len(sys.argv) < 2:
    print("Usage: python script.py <logfile>")
    sys.exit(1)

log_path = sys.argv[1]

result = read_log(log_path)

visualize(result)

plt.show()