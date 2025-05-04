from collections import deque
from typing import Deque

# Bidirectional Search

def bidirectional(graph: Dict[str, List[str]], source: str, goal: str) -> bool:
  def helper(q: Deque[str], visited: set[str]):
    u = q.popleft()
    visited.add(u)

    for v in graph[u]:
      if v in visited:
        continue
      q.append(v)
  
  source_q, goal_q = deque([source]), deque([goal])
  source_visited, goal_visited = set(), set()
  while source_q and goal_q:
    helper(source_q, source_visited)
    helper(goal_q, goal_visited)
    
    for u in graph:
      if u in source_visited and u in goal_visited:
        return True
  return False

bidirectional(
  graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B", "E"],
    "E": ["B", "D", "F"],
    "F": ["C", "E"]
  },
  source='A',
  goal='F'
)
