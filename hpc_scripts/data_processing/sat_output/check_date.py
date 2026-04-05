from goes2go import GOES
from datetime import datetime

g = GOES(satellite=16, product="ABI", domain="C")
# Try fetching a known recent date
g.nearesttime(datetime(2025, 5, 1, 12, 0))   # Does 2025 work
