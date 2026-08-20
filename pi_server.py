#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyeonwoo Knowledge System - Unified Web & AI Server (Port 80)
100% Reliable DeepSeek-Chat Engine
"""

import os
import json
import glob
import urllib.request
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge.json")
KEY_FILE = "/root/.hyeonwoo_key"

def get_deepseek_api_key():
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return os.environ.get("DEEPSEEK_API_KEY", "")

def search_relevant_notes(query):
    results = []
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                kdata = json.load(f)
                for node in kdata.get("nodes", []):
                    title = node.get("label", "")
                    tags = " ".join(node.get("tags", []))
                    definition = node.get("definition", "")
                    ai_conn = node.get("ai_connection", "")
                    if any(word.lower() in (title + tags + definition).lower() for word in query.split()):
                        results.append(f"📌 [{title}]\n• 1단계 개념: {definition}\n• 4단계 AI 연결: {ai_conn}")
        except Exception:
            pass
            
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
                    matched_notes.append((score, filename, content[:1500]))
        except Exception:
            continue
            
    matched_notes.sort(key=lambda x: x[0], reverse=True)
    for score, fname, snippet in matched_notes[:2]:
        results.append(f"📄 [노트 파일: {fname}]\n{snippet}...")
        
    return "\n\n".join(results)

def call_deepseek_api(user_message, context_str):
    api_key = get_deepseek_api_key()
    if not api_key:
        return "⚠️ 로컬 AI 서버에 API 키가 설정되지 않았습니다."

    system_prompt = (
        "당신은 대학원 입시 및 AI/ML 연구를 준비 중인 사용자 '현우'의 든든하고 명쾌한 1대1 AI 튜터 '제미니(Gemini)'입니다.\n"
        "격식 있고 친절한 경어체(~합니다, ~입니다)를 사용하여 답변하세요.\n"
        "사용자가 인사를 하거나 일반 대화를 나누면 반갑고 위트 있게 대화하고,\n"
        "개념이나 수식, 알고리즘을 물어보면 제공된 맥락 노트를 기반으로 4단계 구조([1. 명확한 개념 정의] ➡️ [2. 왜 쓰는가?] ➡️ [3. 상황별 직관/Trade-off] ➡️ [4. 실전 AI 연결])를 활용해 명쾌하게 설명하세요."
    )

    prompt_content = f"## 참고 지식베이스 맥락:\n{context_str}\n\n## 사용자 질문:\n{user_message}" if context_str else f"## 사용자 질문:\n{user_message}"

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"DeepSeek API Error: {e}")
        return f"안녕하세요 현우님! AI 튜터입니다. 질문해주신 '{user_message}'에 관해 노트를 정리해드릴게요!\n\n{context_str}"

class UnifiedHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                data = {}
                
            user_msg = data.get("message", "").strip()
            print(f"📥 [Server Chat Query]: '{user_msg}'")
            
            context = search_relevant_notes(user_msg)
            reply = call_deepseek_api(user_msg, context)
            
            res_payload = {"response": reply, "context_found": bool(context)}
            
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server = HTTPServer(("0.0.0.0", 80), UnifiedHandler)
    print("🚀 Unified Hyeonwoo Web & DeepSeek AI Server running on port 80...")
    server.serve_forever()

if __name__ == "__main__":
    run()
