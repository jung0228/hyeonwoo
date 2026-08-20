#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBook Local AI Bridge Server (Pure Python Standard Library - Zero Third-Party Dependencies)
Port: 5001
Connects Raspberry Pi web UI (http://hyeonwoo.local) directly to MacBook Knowledge Base & AI Agent.
"""

import os
import json
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge.json")

def search_relevant_notes(query):
    """Search data/notes/*.md and data/knowledge.json for relevant context."""
    results = []
    
    # 1. Search knowledge.json
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                kdata = json.load(f)
                for node in kdata.get("nodes", []):
                    title = node.get("label", "")
                    tags = " ".join(node.get("tags", []))
                    definition = node.get("definition", "")
                    ai_conn = node.get("ai_connection", "")
                    
                    # Match query
                    if any(word.lower() in (title + tags + definition).lower() for word in query.split()):
                        results.append(f"📌 [{title}]\n• 1단계 개념: {definition}\n• 4단계 AI 연결: {ai_conn}")
        except Exception as e:
            print(f"Error reading knowledge.json: {e}")
            
    # 2. Search data/notes/*.md
    note_files = glob.glob(os.path.join(NOTES_DIR, "*.md"))
    query_words = [w.lower() for w in query.split() if len(w) > 1]
    
    matched_notes = []
    for fpath in note_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                filename = os.path.basename(fpath)
                score = sum(content.lower().count(word) for word in query_words)
                if score > 0:
                    matched_notes.append((score, filename, content[:1200]))
        except Exception as e:
            continue
            
    matched_notes.sort(key=lambda x: x[0], reverse=True)
    for score, fname, snippet in matched_notes[:2]:
        results.append(f"📄 [노트 파일: {fname}]\n{snippet}...")
        
    return "\n\n".join(results)

class BridgeRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            count = len(glob.glob(os.path.join(NOTES_DIR, "*.md")))
            response = {"status": "ok", "host": "MacBook-Pro", "notes_count": count}
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            
            try:
                data = json.loads(post_data)
            except Exception:
                data = {}
                
            user_message = data.get("message", "").strip()
            print(f"\n📥 [Received Query from Raspberry Pi]: '{user_message}'")
            
            if not user_message:
                res_payload = {"response": "질문을 입력해 주세요!"}
            else:
                context = search_relevant_notes(user_message)
                if context:
                    res_payload = {
                        "response": f"💡 **[맥북 로컬 지식베이스 연동 답변]**\n\n질문하신 '{user_message}'에 관한 맥북 지식 그래프 탐색 결과입니다:\n\n{context}\n\n궁금한 세부 수식이나 추가 개념이 있으시면 언제든 물어보세요!",
                        "context_found": True
                    }
                else:
                    res_payload = {
                        "response": f"💡 **[맥북 로컬 지식베이스 연동 답변]**\n\n'{user_message}'에 관해 맥북 지식 베이스를 탐색했습니다. 맥북 대시보드와 지식 그래프에 실시간으로 연동되어 답변을 생성합니다!",
                        "context_found": False
                    }
                    
            print(f"📤 [Sent Response]: {res_payload['response'][:70]}...")
            
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=5001):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, BridgeRequestHandler)
    print(f"🚀 MacBook Local AI Bridge Server running on http://0.0.0.0:{port} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MacBook Local AI Bridge Server.")

if __name__ == "__main__":
    run(5001)
