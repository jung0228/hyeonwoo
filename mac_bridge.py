#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MacBook Real Google Gemini AI Bridge Server (Pure Python Standard Library)
Port: 5001
Connects Raspberry Pi web UI (http://hyeonwoo.local) directly to MacBook Knowledge Base & Official Google AI Engine.
"""

import os
import json
import glob
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "data", "notes")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "data", "knowledge.json")
KEY_FILE = os.path.expanduser("~/.hyeonwoo_key")

def get_gemini_api_key():
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY", "")

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
                    matched_notes.append((score, filename, content[:1500]))
        except Exception as e:
            continue
            
    matched_notes.sort(key=lambda x: x[0], reverse=True)
    for score, fname, snippet in matched_notes[:2]:
        results.append(f"📄 [노트 파일: {fname}]\n{snippet}...")
        
    return "\n\n".join(results)

def call_gemini_api(user_message, context_str):
    api_key = get_gemini_api_key()
    if not api_key:
        return "⚠️ 로컬 AI 서버에 Google Gemini API 키가 설정되지 않았습니다."

    system_prompt = (
        "당신은 대학원 입시 및 AI/ML 연구를 준비 중인 사용자 '현우'의 든든하고 명쾌한 1대1 원조 Google AI 튜터 '제미니(Gemini)'입니다.\n"
        "격식 있고 친절한 경어체(~합니다, ~입니다)를 사용하여 답변하세요.\n"
        "사용자가 인사를 하거나 일반 대화를 나누면 반갑고 위트 있게 대화하고,\n"
        "개념이나 수식, 알고리즘을 물어보면 제공된 맥락 노트를 기반으로 4단계 구조([1. 명확한 개념 정의] ➡️ [2. 왜 쓰는가?] ➡️ [3. 상황별 직관/Trade-off] ➡️ [4. 실전 AI 연결])를 활용해 명쾌하게 설명하세요."
    )

    full_text = f"{system_prompt}\n\n## 참고 지식베이스 맥락:\n{context_str}\n\n## 사용자 질문:\n{user_message}" if context_str else f"{system_prompt}\n\n## 사용자 질문:\n{user_message}"

    candidate_models = ["gemma-4-31b-it", "gemini-flash-latest"]
    
    for m in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": full_text}]}
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Model {m} failed ({e}), trying next model...")
            continue

    return f"안녕하세요 현우님! 원조 Google Gemini AI 튜터입니다. 질문해주신 '{user_message}'에 관해 맥북 노트를 정리해드릴게요!\n\n{context_str}"

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
            response = {"status": "ok", "host": "MacBook-Pro-Gemini", "notes_count": count, "gemini_ready": bool(get_gemini_api_key())}
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
                gemini_reply = call_gemini_api(user_message, context)
                res_payload = {
                    "response": gemini_reply,
                    "context_found": bool(context)
                }
                    
            print(f"📤 [Official Google Gemini Response Sent]: {res_payload['response'][:80]}...")
            
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
    print(f"🚀 MacBook Official Google Gemini AI Bridge Server running on http://0.0.0.0:{port} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MacBook Local AI Bridge Server.")

if __name__ == "__main__":
    run(5001)
