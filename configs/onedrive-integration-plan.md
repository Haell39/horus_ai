# Plano: Integração OneDrive (Microsoft Graph)

## Objetivo

Definir e documentar um plano passo-a-passo para integrar o backend do Horus com o OneDrive (MS Graph), permitindo autenticação OAuth2, persistência de tokens, seleção de pasta remota e upload de clipes (incluindo uploads em bloco para arquivos grandes).

## Visão geral do fluxo

1. Registrar a aplicação no Azure AD (obter client_id, tenant_id, client_secret). Conceder permissões: Files.ReadWrite.All e offline_access (refresh token).
2. Backend fornece endpoint `/api/v1/onedrive/auth/start` que inicia o fluxo OAuth2 (redireciona para o consent screen do Microsoft). Gera state e salva temporariamente (ex: Redis ou memory) para validação do callback.
3. Azure redireciona para `/api/v1/onedrive/auth/callback?code=...&state=...` do backend. Backend valida state e troca `code` por `access_token` + `refresh_token` (usando MSAL ou requests).
4. Backend persiste tokens com segurança (DB ou arquivo criptografado). Backend também persiste o usuário/tenant associado.
5. Backend expõe endpoints para listar pastas do OneDrive do usuário (ex.: `GET /api/v1/onedrive/folders`), selecionar pasta alvo e para upload (`POST /api/v1/onedrive/upload`).
6. Ao gravar clipe, backend envia para a OneDrive path configurada usando Graph API. Para arquivos grandes, usar Upload Session (chunked upload).

## Requisitos / Pré-requisitos

- Conta Azure com permissões para registrar apps.
- App Registration criado no portal Azure com:
  - client_id (Application (client) ID)
  - tenant_id (Directory (tenant) ID)
  - client_secret configurado (ou certificado)
  - redirect URI: `https://<your-backend>/api/v1/onedrive/auth/callback` (ou http://localhost:8000/... em dev)
- Biblioteca server-side: `msal` (Python) ou usar HTTP requests para token exchange.

## Design detalhado (backend)

Arquivos/Componentes a adicionar:

- `backend/app/api/endpoints/onedrive.py` — router com endpoints:

  - `GET /api/v1/onedrive/auth/start` — gera URL de autorização (state) e redireciona.
  - `GET /api/v1/onedrive/auth/callback` — troca o code por tokens, valida state, persiste tokens e retorna sucesso (página simples ou JSON).
  - `GET /api/v1/onedrive/status` — mostra se a conta está conectada e meta (user, tenant, expiry).
  - `POST /api/v1/onedrive/disconnect` — remove tokens.
  - `GET /api/v1/onedrive/folders?path=...` — lista pastas/children do path (para seleção na UI).
  - `POST /api/v1/onedrive/upload` — recebe arquivo (ou path para arquivo local no servidor) e envia para OneDrive target (configurado).

- `backend/app/core/onedrive_client.py` — wrapper para Microsoft Graph:
  - `get_authorize_url(state)`
  - `acquire_token_by_authorization_code(code)`
  - `acquire_token_by_refresh_token(refresh_token)`
  - `list_children(drive_item_id_or_path)`
  - `create_upload_session(remote_path, size)`
  - `upload_chunked(session_url, file_stream)`

## Token storage

- Armazenar tokens com criptografia: preferir salvar em DB (ex.: SQLite/Postgres) com campo criptografado ou usar `backend/.secrets` com permissões restritas.
- Salvar: access_token, refresh_token, expires_at, scope, tenant, user_id/email.
- Nunca commitar client_secret ou tokens no repo.

## Fluxo de upload

1. Backend verifica se existe OneDrive storage configurado (em `storage_config`): mode=onedrive e target remote path/id.
2. Quando gerar um clipe, em vez de copiar para local-only, backend chama `onedrive_client.create_upload_session()` para o caminho remoto desejado.
3. Se o arquivo for pequeno (< 4 MB) pode usar `PUT /me/drive/root:/path:/content`.
4. Para arquivos maiores, usar `POST /me/drive/items/{item-id}:/createUploadSession` e depois `PUT` nos ranges (ver docs MS Graph chunked upload).
5. Após upload completo, persistir o link público (ou sharing link) no campo `evidence` da ocorrência, por exemplo `evidence.clip_url = 'https://...onedrive...'`.

## Considerações de UI (frontend)

- Botões/UX:
  - ‘Conectar OneDrive’ — abre uma nova janela apontando para `GET /api/v1/onedrive/auth/start`.
  - Após sucesso, UI pode requisitar `GET /api/v1/onedrive/folders` para permitir selecionar pasta alvo.
  - Exibir status da conexão (conectado, expirado, precisa reconectar).
  - A opção de desconectar revoga tokens localmente.

## Segurança

- Guardar `client_secret` em variáveis de ambiente no backend: `ONEDRIVE_CLIENT_ID`, `ONEDRIVE_CLIENT_SECRET`, `ONEDRIVE_TENANT_ID`.
- Validar `state` no callback para prevenir CSRF.
- Implementar escopo mínimo `Files.ReadWrite.All offline_access` e explicar ao usuário por que esse scope é necessário.
- Implementar permissões/whitelist se necessário (por ex: aceitar apenas uploads para uma pasta root gerenciada).

## Dependências sugeridas

- Python: `msal` (Microsoft Authentication Library for Python) para troca de token e refresh handling.
- Para uploads chunked: usar `requests` (ou httpx) com streaming.
- Para Node/Electron flows (se for fazer client side): `@azure/msal-browser` ou `msal`.

## Edge cases / erros a tratar

- Refresh token expirado ou revogado → falha na tentativa de upload; notificar usuário e solicitar reconexão.
- Limites de quota / rate-limits da Graph API → aplicar retries com backoff exponencial e log.
- Upload interrompido → retomar via upload session se suportado (manter session URL temporariamente).
- Arquivos com nomes inválidos ou caminhos com caracteres especiais → normalizar e escapar nomes.

## Exemplos de endpoints e payloads

- Start auth (backend):
  GET /api/v1/onedrive/auth/start → redirect (302) para Microsoft consent URL

- Callback (backend):
  GET /api/v1/onedrive/auth/callback?code=...&state=...
  -> backend troca token e responde com pagina de sucesso ou redireciona para frontend com status.

- Upload (backend):
  POST /api/v1/onedrive/upload
  Headers: Authorization: Bearer <server-admin-token> (se proteger endpoint)
  Body (multipart/form-data) or JSON pointing to server-side file path.

## Checklist de testes manuais

- Registrar app no Azure e confirmar que o redirect URI funciona em dev.
- Fluxo completo: clicar Connect → consent → backend troca tokens → UI mostra conectado.
- Selecionar pasta no OneDrive via endpoint e selecionar como destino.
- Fazer upload de clipe pequeno e verificar chega ao OneDrive.
- Fazer upload grande (>10–50MB) usando chunked upload e verificar integridade.
- Simular refresh token expirado e validar que a reconexão é solicitada.

## Notas finais

Esta integração exige cuidado com credenciais e tokens. Para desenvolvimento vale usar um tenant/dev app e testar exaustivamente antes de aplicar em produção com contas de usuários reais. Quando for implementar eu posso gerar o scaffold backend (`onedrive.py`, `onedrive_client.py`), exemplos de chamadas e testes automáticos minimalistas.
