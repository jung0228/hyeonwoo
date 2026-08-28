#!/usr/bin/env python3
"""
지식 그래프 무결성 검증 및 통계 스크립트
- 노드와 엣지의 일관성 검사 (Critical)
- 고립 노드(0-degree) 검사 (Warning)
- 마크다운 노트 파일 작성 현황 추적 (Info/Pending)
- 카테고리 유효성 검사 (Warning)
- 학습 세션 유효성 검사 (Warning)
"""

import json
import os
import sys

ALLOWED_CATEGORIES = {
    "Generative",
    "Architecture",
    "Language Model",
    "Multimodal",
    "Training",
    "RL",
    "Math & Stats",
    "Systems",
    "Algorithm",
    "Math",
    "Math Problems",
    "DeepLearning",
    "Career"
}

def verify():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_path = os.path.join(base_dir, "data", "knowledge.json")
    
    if not os.path.exists(knowledge_path):
        print(f"❌ Error: {knowledge_path} not found.")
        sys.exit(1)
        
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    sessions = data.get("sessions", [])
    
    node_map = {n["id"]: n for n in nodes}
    critical_errors = []
    warnings = []
    missing_notes = []
    existing_notes_count = 0
    
    print(f"==========================================")
    print(f"📊 [정현우의 지식 지도 데이터 무결성 검증]")
    print(f"==========================================")
    print(f"• 총 노드 수: {len(nodes)}개")
    print(f"• 총 엣지 수: {len(edges)}개")
    print(f"• 등록된 학습 세션: {len(sessions)}개")
    
    # 1. Node validation & Note checking
    for n in nodes:
        nid = n.get("id")
        if not nid:
            critical_errors.append(f"ID가 없는 노드가 발견되었습니다: {n}")
            continue
            
        cat = n.get("category")
        if cat not in ALLOWED_CATEGORIES:
            warnings.append(f"노드 '{nid}'의 카테고리가 비표준입니다: '{cat}'")
            
        note_file = n.get("note", f"data/notes/{nid}.md")
        full_note_path = os.path.join(base_dir, note_file)
        if os.path.exists(full_note_path):
            existing_notes_count += 1
        else:
            missing_notes.append((nid, n.get("label", nid), note_file))
            
    # 2. Edge validation
    connected_nodes = set()
    for i, e in enumerate(edges):
        s = e.get("source")
        t = e.get("target")
        if s not in node_map:
            critical_errors.append(f"엣지 #{i}의 source '{s}' 노드가 존재하지 않습니다.")
        else:
            connected_nodes.add(s)
            
        if t not in node_map:
            critical_errors.append(f"엣지 #{i}의 target '{t}' 노드가 존재하지 않습니다.")
        else:
            connected_nodes.add(t)
            
    # 3. Orphan node check
    orphan_nodes = [nid for nid in node_map if nid not in connected_nodes]
    if orphan_nodes:
        warnings.append(f"연결선(엣지)이 없는 고립 노드 ({len(orphan_nodes)}개): {orphan_nodes}")
        
    # 4. Sessions validation
    for s in sessions:
        date = s.get("date")
        topics = s.get("topics", [])
        for t in topics:
            if t not in node_map:
                warnings.append(f"세션 날짜 '{date}'에 정의되지 않은 노드 참조: '{t}'")
                
    # 5. Note completion rate
    total_nodes = len(nodes)
    completion_rate = (existing_notes_count / total_nodes * 100) if total_nodes > 0 else 0
    print(f"• 상세 노트 작성 현황: {existing_notes_count}/{total_nodes} ({completion_rate:.1f}%)\n")

    # 6. 고아 파일 검사 (notes 디렉토리에 있지만 어떤 노드에도 연결되지 않은 .md 파일)
    notes_dir = os.path.join(base_dir, "data", "notes")
    orphan_files = []
    convention_hits = []  # note 필드는 없지만 파일명 규칙(data/notes/<node_id>.md)으로 존재하는 파일
    if os.path.isdir(notes_dir):
        for root, dirs, files in os.walk(notes_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                rel_path = os.path.relpath(os.path.join(root, fname), base_dir)
                # 어떤 노드의 note 필드가 이 파일을 가리키는지 확인
                referenced = any(
                    n.get("note") and os.path.normpath(n["note"]) == os.path.normpath(rel_path)
                    for n in nodes
                )
                if referenced:
                    continue
                # 파일명 규칙(data/notes/<node_id>.md)과 일치하는 노드가 있는지 확인
                note_id = fname[:-3]
                matching_node = next((n for n in nodes if n["id"] == note_id), None)
                if matching_node and not matching_node.get("note"):
                    convention_hits.append((matching_node["id"], matching_node.get("label", note_id), rel_path))
                else:
                    orphan_files.append(rel_path)

    if convention_hits:
        print(f"💡 [note 필드 미지정 - 파일명 규칙으로 존재] ({len(convention_hits)}개):")
        for nid, lbl, nf in convention_hits:
            print(f"   • {lbl} (`{nf}`) — knowledge.json의 해당 노드에 `\"note\": \"{nf}\"` 추가 권장")
        print()

    if orphan_files:
        print(f"🗂️ [고아 노트 파일 - 어떤 노드에도 연결 안 됨] ({len(orphan_files)}개):")
        for nf in orphan_files:
            print(f"   - {nf}")
        print()

    # Results
    if critical_errors:
        print(f"❌ [치명적 오류 - 그래프 연결 깨짐] ({len(critical_errors)}개):")
        for err in critical_errors:
            print(f"   - {err}")
        sys.exit(1)
    else:
        print("✅ [그래프 무결성] 노드 ID 및 엣지 연결이 100% 완벽하게 유효합니다.")
        
    if warnings:
        print(f"\n⚠️ [주의/알림 사항] ({len(warnings)}개):")
        for w in warnings:
            print(f"   - {w}")
            
    if missing_notes:
        print(f"\n📝 [작성 대기 중인 상세 노트] ({len(missing_notes)}개):")
        for nid, lbl, nf in missing_notes[:8]:
            print(f"   • {lbl} (`{nf}`)")
        if len(missing_notes) > 8:
            print(f"   • ... 외 {len(missing_notes)-8}개")
            
    print("\n✨ 지식 지도 데이터 검증 완료!")

if __name__ == "__main__":
    verify()
