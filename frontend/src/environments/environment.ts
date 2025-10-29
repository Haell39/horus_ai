// frontend/src/environments/environment.ts
// (Este arquivo é para desenvolvimento)

export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api/v1', // A URL da nossa API FastAPI
  // URL do stream ao vivo (HLS .m3u8 preferencial). Deixe vazio para usar player local.
  liveStreamUrl:
    'srt://168.90.225.116:6052?mode=caller&latency=4000&transtype=live&passphrase=yKz585@353&pbkeylen=16',
  // Base do backend para construir URLs HTTP (usado para HLS gerado no backend)
  backendBase: 'http://localhost:8000',
};
