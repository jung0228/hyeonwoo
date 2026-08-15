# 알고리즘 핵심 패턴

카테고리: Algorithm  
자신감: ⭐⭐⭐ (중급)  
마지막 복습: 2026-08-08

> 코테 90분 3문제 Python 기준. 아래 패턴 중 하나 이상 복합 적용.

---

## 1. Sliding Window

언제: 연속 구간에서 조건을 만족하는 최장/최단 길이

```python
def longest_at_most_k(items, k):
    count = {}
    left = 0
    answer = 0

    for right, item in enumerate(items):
        count[item] = count.get(item, 0) + 1

        while len(count) > k:
            old = items[left]
            count[old] -= 1
            if count[old] == 0:
                del count[old]
            left += 1

        answer = max(answer, right - left + 1)

    return answer
```

> Prefix Sum과 비교: Sliding Window는 가변 크기 창, Prefix Sum은 고정 구간 합

---

## 2. Prefix Sum + Dictionary

언제: 합이 target인 연속 부분배열 개수

```python
def count_target_subarrays(nums, target):
    seen = {0: 1}
    prefix = 0
    answer = 0

    for num in nums:
        prefix += num
        answer += seen.get(prefix - target, 0)
        seen[prefix] = seen.get(prefix, 0) + 1

    return answer
```

> 핵심: 현재 누적합이 `prefix`라면, 이전에 `prefix - target`이 나온 횟수만큼 정답에 더함

---

## 3. Heap + Interval

언제: 동시에 필요한 최소 자원 수 (강의실, 서버 등)

```python
import heapq

def min_resources(intervals):
    intervals.sort()
    heap = []  # 사용 중인 작업의 종료 시간

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)  # 가장 빨리 끝나는 것 재활용
        else:
            heapq.heappush(heap, end)    # 새 자원 추가

    return len(heap)
```

---

## 4. Binary Search on Answer (정답 이분탐색)

언제: 정답 범위를 알고, 특정 값이 가능한지 O(n)으로 판정 가능할 때

```python
def minimum_capacity(files, days):
    def possible(capacity):
        used_days = 1
        current = 0
        for size in files:
            if current + size > capacity:
                used_days += 1
                current = 0
            current += size
        return used_days <= days

    left, right = max(files), sum(files)
    while left < right:
        mid = (left + right) // 2
        if possible(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## 5. BFS Grid (최단 경로)

```python
from collections import deque

def shortest_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0)])
    visited = {start}
    dirs = ((1,0),(-1,0),(0,1),(0,-1))

    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] != 1 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    return -1
```

> A*와의 관계: BFS는 h(n)=0인 A*. 거리 추정치가 있으면 A*로 가속 가능.

---

## 6. Topological Sort (Kahn's Algorithm)

언제: 선후 관계가 있는 작업 순서, DAG 순서 정렬

```python
from collections import deque

def topological_sort(n, edges):
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    for before, after in edges:
        graph[before].append(after)
        indegree[after] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    return order if len(order) == n else None  # None이면 cycle 존재
```

---

## 7. Dynamic Programming

인접하지 않은 원소의 최대 합 예시:

```python
from functools import lru_cache

def max_non_adjacent_sum(values):
    @lru_cache(None)
    def dp(i):
        if i >= len(values):
            return 0
        skip = dp(i + 1)
        take = values[i] + dp(i + 2)
        return max(skip, take)
    return dp(0)
```

> `@lru_cache`는 같은 i를 중복 계산하지 않음. 없으면 지수 시간.

---

## 시험 직전 우선순위

1. 중첩 dict/list + XML → DataFrame 변환
2. pandas: filtering, groupby, sort, fillna
3. PyTorch: tensor dimension, `model.eval()`, `no_grad()`, `argmax(dim=1)`
4. 슬라이딩 윈도우, prefix sum, heap, binary search, BFS, topo sort
5. 빈 입력, 중복, `None`, 매우 큰 입력 처리

---

## 체크리스트

- [x] Sliding Window 패턴 구현
- [x] Prefix Sum + Hash 패턴
- [x] Heap + Interval 스케줄링
- [x] Binary Search on Answer
- [x] BFS Grid 최단 경로
- [x] Topological Sort (Kahn)
- [x] DP with lru_cache
- [ ] Dijkstra 구현 (heap + visited)
- [ ] Union-Find (DSU) 구현
