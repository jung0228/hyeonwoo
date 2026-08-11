# CPU Pipeline & Hazard

**카테고리**: Systems  
**자신감**: ⭐⭐⭐ (중급)  
**마지막 복습**: 2026-08-10

---

## 한 문장 요약

CPU Pipeline은 IF→ID→EX→MEM→WB 5단계를 동시에 진행해 throughput을 높이지만, **Hazard(충돌)**가 발생하면 stall이나 forwarding으로 해결한다.

---

## 5단계 파이프라인

```
IF  → ID  → EX  → MEM → WB
     (다음 명령어)
          (다다음 명령어)
```

- **IF**: Instruction Fetch
- **ID**: Decode + Register Read
- **EX**: ALU / Address 계산
- **MEM**: Data Memory 접근
- **WB**: Register Write-back

> Cycle time = 가장 느린 stage가 결정

---

## Hazard 세 종류

```
Structural :: 같은 hardware resource를 동시에 요구
Data       :: 필요한 operand가 아직 준비되지 않음
Control    :: branch의 다음 PC가 미정
```

### RAW (Read After Write) — 진짜 dependency

```asm
ADD R1, R2, R3   ← R1에 씀
SUB R4, R1, R5   ← R1을 읽어야 함
```

→ **Forwarding**: EX→EX, MEM→EX 경로로 결과 전달  
→ **Load-use**: 메모리에서 읽는 경우 1 cycle stall 불가피

### WAR / WAW — 이름 충돌 (Out-of-Order에서만 문제)

```
WAR :: 앞 read → 뒤 write (OoO에서 뒤가 먼저 쓰면 앞이 잘못된 값 읽음)
WAW :: 두 write가 같은 레지스터 (늦은 write가 먼저 반영될 수 있음)
```

→ **Register Renaming**: 새 physical register에 매핑해서 해결  
→ RAW는 renaming으로 제거 불가 (진짜 dependency)

### Out-of-Order (OoO) Execution

앞 instruction이 기다리는 동안 **독립적인 뒤 instruction을 먼저 실행**  
→ Reorder Buffer(ROB)가 program order로 commit 보장

---

## Stall vs Bubble

```
Stall  :: instruction을 기다리게 함 (파이프라인 멈춤)
Bubble :: stall의 결과로 생긴 빈 cycle (NOP)
```

---

## 체크리스트

- [x] 5단계 이름과 역할 설명
- [x] RAW / WAR / WAW 구분
- [x] Forwarding 작동 원리
- [x] Load-use hazard 1-cycle stall 이유
- [x] OoO + ROB 설명
- [ ] Branch prediction (static / dynamic) 설명
- [ ] Speculative execution 설명
