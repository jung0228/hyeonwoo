# OS / I/O / 인터럽트

**카테고리**: Systems  
**자신감**: ⭐⭐⭐ (중급)  
**마지막 복습**: 2026-08-09

---

## 개념 구분

```
Hardware Interrupt
 : device·timer가 외부에서 비동기적으로 발생
 : CPU가 현재 instruction 완료 후 handler로 이동

Exception
 : 현재 instruction 때문에 동기적으로 발생
 : page fault, divide-by-zero, invalid opcode

System Call
 : user program이 kernel service를 의도적으로 요청 (동기)
 : x86: syscall / ARM: SVC / RISC-V: ecall
```

### Driver vs Handler

```
Driver  : device를 관리하는 kernel code 전체
Handler : interrupt가 오면 실행되는 driver 내부 함수
```

---

## SSD Read 전체 흐름

```
1. Application → read() system call
2. CPU: user mode → kernel mode 전환
3. VFS + file system: inode → block mapping 확인
4. Storage driver: controller에 I/O 요청
5. Kernel: RAM buffer 확보
6. Controller의 DMA engine: SSD data → RAM buffer 직접 기록
7. 전송 완료 → hardware interrupt 발생
8. IRQ handler: 완료 확인
9. 기다리던 process를 깨움
10. read() 반환 → application 복귀
```

### 역할 한 줄 정리

```
Controller : SSD를 실제로 제어하는 hardware 관리자
DMA        : device ↔ RAM 간 data를 CPU 대신 전송
Buffer     : RAM 안에 미리 확보한 임시 data 공간
Interrupt  : I/O 완료를 CPU에 알리는 신호
Driver     : kernel 요청을 controller 명령으로 변환 + 완료 처리
```

---

## File System 연계

```
file name
→ directory entry → inode number
→ inode: metadata + block mapping
→ offset / block size → logical block 계산
→ direct/indirect pointer로 storage block 탐색
```

- **inode**: filename 제외한 metadata + data block 위치
- **Direct pointer**: data block 직접 가리킴
- **Indirect**: block 주소표를 1~3단계로 가리킴

### Crash Consistency

```
Journaling
 : 변경 log + commit을 먼저 durable하게 기록 → replay로 복구

Copy-on-Write (CoW)
 : 기존 block 덮지 않고 새 block에 완성 후 root pointer 교체
 : ZFS, Btrfs, APFS
```

---

## 체크리스트

- [x] Interrupt / Exception / System Call 구분
- [x] SSD read 전체 흐름 10단계 설명
- [x] DMA 역할 (CPU 대신 전송)
- [x] Journaling vs CoW 비교
- [ ] Interrupt handling 중 다른 interrupt 처리 (nested interrupt)
- [ ] softirq vs tasklet vs workqueue 차이 (Linux)
