import pstats
from pstats import SortKey

top_number = 100
file_name = 'time_summery'
p = pstats.Stats(file_name)
# print(f"{'*' * 10 }Profile summery of {file_name}{ '*' * 10}")
p.strip_dirs().sort_stats(-1).print_stats()
print(f"{'*' * 10}Time consuming top {top_number}  from {file_name}{'*' * 10}")
p.sort_stats(SortKey.TIME).print_stats(top_number)
