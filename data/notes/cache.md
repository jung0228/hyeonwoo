# Cache 구조

카테고리: Systems  
자신감: ⭐⭐⭐ (중급)  
마지막 복습: 2026-08-10


## 한 문장 요약

Cache는 메모리 계층 구조에서 DRAM보다 빠른 임시 저장소로, Tag | Set Index | Block Offset으로 주소를 분해해서 데이터를 찾는다.


## 주소 분해

```
Physical Address = Tag | Set Index | Block Offset
```

- Block Offset: cache line 안의 byte 위치
- Set Index: 어느 set을 확인할지 선택
- Tag: 그 set 안의 여러 block 중 어떤 것인지 식별

### Hit / Miss

```
Hit  : valid bit = 1 AND tag 일치 → 데이터 사용
Miss : lower cache / DRAM에서 whole line 가져옴
```


## Direct vs Set Associative

```
Direct-mapped (1-way)
 : 각 block이 딱 1개의 cache line에만 들어갈 수 있음
 : 빠르고 단순하지만 conflict miss가 큼

N-way Set Associative
 : 같은 set index의 N개 way 중 하나에 저장
 : conflict miss 감소, tag 비교·replacement 복잡
```

> `conflict miss` = 서로 다른 데이터가 같은 set을 두고 경쟁해서 발생하는 miss


## Cache Line 크기 트레이드오프

| 크게 하면 | 작게 하면 |
|---|---|
| Spatial locality 활용 | 공간 낭비 감소 |
| 연속 접근 miss 감소 | 더 많은 line 보관 가능 |
| 불필요한 byte 전송, bandwidth 낭비 | 연속 data도 여러 번 miss 가능 |
| line 수 감소, miss penalty 증가 | tag overhead 증가 |


## Write 정책

```
Write-through  : 캐시 + 메모리 동시에 씀 → 일관성 쉽지만 느림
Write-back     : 캐시에만 씀, evict 시 메모리 업데이트
Write buffer   : write-through에서 queue로 CPU 대기 줄임
```


## False Sharing

서로 다른 core가 다른 변수를 수정해도,  
두 변수가 같은 cache line에 있으면  
→ line 전체가 core 사이에서 invalidation·이동 발생  
→ 성능 급감 (실제로 공유 안 하는데도)

해결: 변수 padding으로 cache line에 혼자 있게 하기


## Pipeline과의 관계

- Load-use hazard: 메모리 접근 → cache miss → stall 길어짐
- L1 cache: pipeline의 MEM stage와 직결
- TLB: 가상 주소 → 물리 주소 translation의 cache


## 체크리스트

- [x] Tag/Index/Offset 분해 설명
- [x] Hit/Miss 판정 과정
- [x] Direct vs Set-associative 비교
- [x] Cache line 크기 트레이드오프
- [x] False sharing 설명
- [ ] VIPT cache (Virtual Index, Physical Tag) 설명
- [ ] 3C miss (Compulsory/Capacity/Conflict) 분류
