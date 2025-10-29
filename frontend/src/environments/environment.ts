// frontend/src/environments/environment.ts
// (Este arquivo é para desenvolvimento)

export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api/v1', // A URL da nossa API FastAPI
  // URL do stream ao vivo (HLS .m3u8 preferencial). Deixe vazio para usar player local.
  liveStreamUrl: '',
  // Identificador lógico do stream configurado no backend (.env)
  streamId: 'globo',
  // Base do backend para construir URLs HTTP (usado para HLS gerado no backend)
  backendBase: 'http://localhost:8000',
};
