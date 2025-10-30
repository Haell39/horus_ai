# Ambiente (.env) — Horus AI

Este documento explica como usar o `backend/.env.example` para criar seu `backend/.env` local (não comitar).

1. Copiar o exemplo

```powershell
Copy-Item backend\.env.example backend\.env
notepad backend\.env   # edite no Windows com seus valores
```

2. Variáveis importantes (descrição curta)

- `DATABASE_URL` — URL do Postgres (ex.: postgresql://user:pass@localhost:5432/dbname)
- `SRT_STREAM_URL_GLOBO` — exemplo SRT com passphrase (mantenha secreto)

Tuning de vídeo (detecção)

- `VIDEO_VOTE_K` — número de frames consecutivos (K) necessários para considerar um evento. Ex.: `VIDEO_VOTE_K=3`.
- `VIDEO_MOVING_AVG_M` — janela M para média móvel de confiança. Ex.: `VIDEO_MOVING_AVG_M=5`.
- `VIDEO_THRESH_<CLASSE>` — limiar por classe. Ex.: `VIDEO_THRESH_BORRADO=0.7`.
- `VIDEO_THRESH_DEFAULT` — fallback quando não há threshold por classe.

3. Recomendações

- NÃO comite `backend/.env`.
- Use `backend/.env.example` como fonte de verdade para quais chaves existem.
- Reinicie o backend após editar `.env`.

4. Exemplo mínimo no `backend/.env`

```
DATABASE_URL=postgresql://globo_user:senha123@localhost:5432/globo_mvp
VIDEO_VOTE_K=3
VIDEO_MOVING_AVG_M=5
VIDEO_THRESH_BORRADO=0.7
VIDEO_THRESH_BLOCO=0.7
```
