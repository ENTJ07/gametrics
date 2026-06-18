# Gametrics — 배포 가이드

**구조**: Netlify(정적 사이트) + Render(파이썬 API). Netlify가 `/api/*`를 Render로 프록시 → 브라우저엔 same-origin(CORS 불필요).

배포에 필요한 파일은 모두 커밋됨: `web/`(사이트), `data/artifacts.joblib`(채점 모델), `server.py`/`score.py`/`riot_client.py`(백엔드), `requirements.txt`, `render.yaml`, `netlify.toml`. `.env`와 raw 데이터는 제외(gitignore).

---

## 1. GitHub에 푸시
```bash
# git init + 첫 커밋은 이미 완료됨
# github.com 에서 새 repo 생성 (public 또는 private 둘 다 가능)
git remote add origin https://github.com/<당신>/gametrics.git
git branch -M main
git push -u origin main
```

## 2. 백엔드 → Render
1. https://render.com → **New** → **Web Service** → GitHub repo 연결
2. `render.yaml` 자동 감지(Blueprint). 수동이면:
   - Runtime: **Python**
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. **Environment → Add**: `RIOT_API_KEY` = (당신의 personal key)  ← 절대 커밋 금지, 여기에만
4. Deploy → URL 확인: `https://<이름>.onrender.com`
5. 테스트: `https://<이름>.onrender.com/api/evaluate?name=권유진&tag=KR1`

## 3. 프록시 연결
- `netlify.toml`의 `to =` URL을 **실제 Render URL**로 교체 → commit + push
```toml
  to = "https://<당신의-render-이름>.onrender.com/api/:splat"
```

## 4. 정적 → Netlify
1. https://netlify.com → **Add new site** → **Import from Git** → repo 선택
2. Build command: (비움) · **Publish directory: `web`** (netlify.toml에 이미 설정됨)
3. Deploy → 사이트: `https://<이름>.netlify.app`

## 5. 검증
- 리더보드 + 수집된 선수 조회: 즉시
- 실시간 조회(미수집 선수): Netlify→Render 프록시. **유휴 후 첫 호출 ~30초**(Render 무료티어 콜드스타트)

---

## 운영 메모
- **키 보안**: `RIOT_API_KEY`는 Render 환경변수에만. 코드/커밋에 절대 넣지 말 것.
- **데이터 갱신**: 파이프라인 재실행 → `python export_web.py` + `python build_artifacts.py` → `web/data.json`·`data/artifacts.joblib` 커밋·푸시 → 양쪽 자동 재배포.
- **레이트리밋**: personal key 20/s·100/2min(서버가 자동 준수).
- **신패치**: 새 패치 데이터 수집 후 `PATCH`(server.py)·`START` 갱신 + 재학습.
