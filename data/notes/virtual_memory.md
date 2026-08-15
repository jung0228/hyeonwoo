# Virtual Memory & Paging

카테고리: Systems  
자신감: ⭐⭐⭐ (중급)  
마지막 복습: 2026-08-09

---

## 한 문장 요약

Virtual Memory는 프로세스가 연속된 큰 메모리 공간을 가지는 것처럼 추상화하고, MMU + Page Table + TLB가 가상→물리 주소를 변환한다.

---

## 주소 변환 흐름

```
Virtual Address = VPN | Offset
         ↓
      MMU가 TLB 확인
         ↓
   TLB Hit? → PPN + 같은 Offset → Physical Address
         ↓ Miss
   Page Table Walk (RAM에서)
         ↓
   PTE present? → TLB 갱신 후 접근
         ↓ Not Present
      Page Fault → OS가 처리
```

### PTE 구성 요소

```
PTE: PPN | present | dirty | accessed | permission(r/w/x)
```

- TLB: 최근 VPN→PPN mapping cache (miss 시 page table walk)

---

## Page Fault 처리

정상 주소, 하지만 page가 RAM에 없을 때:

1. OS가 SSD/swap에서 해당 page를 RAM frame으로 가져옴
2. PTE + TLB 갱신
3. 원래 instruction 하나만 재실행 (program 전체 재시작 ❌)

잘못된 주소 / 권한 위반 시:
→ signal 발생 또는 프로세스 종료 (복구 불가)

---

## Paging 장단점

```
장점
 - External fragmentation 제거
 - Protection & Sharing (PTE permission)
 - Demand paging (필요한 page만 RAM에)

단점
 - Internal fragmentation (page 내 빈 공간)
 - 큰 page table (64-bit: 수 GB)
 - Translation + page-fault 비용
```

---

## Page Table 구조들

```
Hierarchical Page Table
 : sparse virtual address → 사용 영역의 하위 table만 생성
 : x86-64: 4단계 (PML4 → PDPT → PD → PT)

Inverted Page Table
 : physical frame당 entry 하나
 : table 크기 = physical memory 크기
 : PID+VPN search가 느림 → hash로 보완
```

---

## Swap & Thrashing

```
Swapping  : page를 RAM ↔ disk swap area로 이동
Thrashing : working set > RAM → page fault/swap만 반복
```

> Thrashing 해결: working set 모니터링, 프로세스 줄이기, RAM 증설

---

## Cache와의 관계

- TLB = 주소 변환 결과의 cache
- Page table walk = cache miss 상황의 fallback
- Memory-mapped I/O = file을 virtual address에 매핑

---

## 체크리스트

- [x] VPN → PPN 변환 흐름 설명
- [x] TLB hit/miss 과정
- [x] Page fault 처리 (instruction 재실행)
- [x] Hierarchical page table 필요성
- [x] Thrashing 원인과 해결
- [ ] x86-64 4단계 page table 구조 그리기
- [ ] ASID (Address Space ID)로 TLB flush 줄이는 방법
