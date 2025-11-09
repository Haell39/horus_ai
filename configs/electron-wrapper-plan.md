# Plano: Electron wrapper (esboço de implementação)

## Objetivo

Documentar um plano claro e reutilizável para implementar, testar e empacotar um wrapper Electron que permita ao usuário escolher uma pasta local nativamente e persistir essa escolha no backend via `/api/v1/admin/storage-config`.

## Resumo (alto nível)

- Criar uma mini-app Electron que carrega a UI Angular (dev: `http://localhost:4200`, prod: build estático).
- Expor na `preload` uma API segura `window.electronAPI.selectFolder()` que abre o diálogo nativo e devolve o caminho absoluto selecionado.
- O processo `main` faz o POST ao endpoint admin do backend para gravar a configuração (`mode: local`, `local_path: <caminho>`).
- A UI detecta se está dentro do Electron e mostra um botão opcional “Selecionar Pasta (Desktop)” que chama essa API.

## Por que usar Electron

- Browsers não expõem caminhos absolutos por segurança; Electron permite diálogos nativos e acesso a paths locais.
- Melhora a experiência do usuário final (UX nativa).

## Onde colocar os arquivos

- `electron/` (nova pasta na raiz)
  - `main.js` — processo principal do Electron
  - `preload.js` — expose API via contextBridge
  - `package.json` (opcional) — config local do electron
- Atualizar `frontend/package.json` com scripts para dev/pack

## Dependências (sugestão mínima)

- `electron` (dev dependency)
- `electron-builder` (se vai empacotar instalador)
- `node-fetch` (opcional; Node 18+ tem fetch global)

## Estrutura mínima dos arquivos

1. electron/main.js (esqueleto)

- Criar BrowserWindow, carregar `http://localhost:4200` em dev ou `file://.../index.html` em prod.
- Registrar IPC handler `ipcMain.handle('select-folder', async () => { ... })` que executa `dialog.showOpenDialog({ properties: ['openDirectory'] })`.
- Quando o usuário escolher um diretório, o main fará um POST ao backend `/api/v1/admin/storage-config` com payload `{ mode: 'local', local_path: '<caminho>' }`. Incluir headers com token se necessário.

2. electron/preload.js

- Usar `contextBridge.exposeInMainWorld('electronAPI', { selectFolder: () => ipcRenderer.invoke('select-folder') })`.
- Não expor outras APIs para o renderer.

3. frontend (Angular)

- Detectar `window.electronAPI?.selectFolder`.
- Adicionar botão `Selecionar Pasta (Desktop)` que chama `await window.electronAPI.selectFolder()`; exibir resultado (sucesso/erro). Não fazer POST do renderer; deixe o `main` fazer a chamada ao backend.
- Fallback: manter o seletor atual (File System Access API / webkitdirectory) para navegadores.

## Fluxo de interação (dev)

1. Rode backend (uvicorn) em `http://localhost:8000`.
2. Rode frontend (`npm start` / `ng serve`), disponível em `http://localhost:4200`.
3. Rode electron em modo dev (script `electron:dev`): o electron abre e carrega `http://localhost:4200`.
4. Usuário clica em “Selecionar Pasta (Desktop)”.
5. `preload` solicita `select-folder` ao `main` → main abre `dialog.showOpenDialog` → retorna caminho.
6. `main` faz POST para `http://localhost:8000/api/v1/admin/storage-config` com JSON `{ mode: 'local', local_path: '<caminho>' }`.
7. Backend persiste o config; o `main` retorna sucesso/erro ao renderer; renderer mostra o resultado ao usuário.

## Segurança e hardening

- NÃO exponha o endpoint admin sem autenticação em produção. Recomendações:
  - Adicionar checagem de origem (apenas localhost) e/ou token (header `X-ADMIN-TOKEN`).
  - Se usar token, armazenar em `process.env.ADMIN_TOKEN` no Electron (ou pedir ao usuário na primeira execução).
- Fazer o POST a partir do processo `main` evita CORS/limitações de `file://` e não expõe credenciais ao renderer.
- Validar o caminho no backend: checar que o path existe e é um diretório, aplicar allowlist se necessário.

## Scripts npm sugeridos (frontend/package.json)

- `"electron:dev": "concurrently \"ng serve\" \"electron .\""` (usar concurrently) — simplifica rodar tudo em dev.
- `"electron:build": "ng build --configuration production && electron-builder"` — build e empacota (configurar electron-builder).

## Notas sobre empacotamento (electron-builder)

- Configurar `build` no `package.json` (appId, directories, files, win/nsis config).
- Testar installadores e atualizações automáticas separadamente.

## Testes e validação

- Teste manual: selecionar pastas em Windows (D:\...), Mac (/Users/...), Linux (/home/...), verificar que backend recebe `local_path` e responde.
- Teste de erro: caminho inválido, sem permissão, backend inacessível — o main deve retornar erro claro ao renderer.
- Integração: confirme que clipes gravados pelo backend aparecem em `/clips` mesmo quando `local_path` foi alterado.

## Coisas a documentar na entrega

- Como rodar em dev (comandos, requisitos: Node version, electron version, ffmpeg, backend running).
- Como construir/empacotar (electron-builder notes).
- Onde colocar token admin (env var) e como proteger endpoint.
- Checklist de QA (permissões, antivírus, caminhos com espaços, usuários com caracteres unicode).

## Alternativas e trade-offs

- Electron: melhor UX, distribuição necessária; aumenta complexidade de empacotamento.
- Agent/daemon: mais adequado para infra centralizada (roda separado e faz sync), exige instalação de serviço.
- PowerShell helper: mais simples (já implementado), bom para dev/ops, não é seamless para usuários finais.

## Checklist mínimo para lançar (MVP Electron)

- [ ] `electron/main.js` + `preload.js` prontos e seguros
- [ ] Botão UI integrado com fallback
- [ ] Script `electron:dev` e instruções dev
- [ ] Segurança: token ou whitelist de localhost documentada
- [ ] Testes manuais passados em Windows (pelo menos)
- [ ] Documentação finalizada nesta pasta `configs/` apontando para os passos de empacotamento

## Observações finais

Este documento é um roteiro prático para quando decidirmos implementar o wrapper Electron. Ele prioriza uma implementação simples e segura (o `main` faz o POST; o renderer só pede ao main). Se quiser, quando chegar a hora eu aplico o scaffold e os scripts automaticamente.

-- fim do plano
