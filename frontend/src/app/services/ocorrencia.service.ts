// frontend/src/app/services/ocorrencia.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// 1. Importar o environment e a nova interface
import { environment } from '../../environments/environment';
import { Ocorrencia } from '../models/ocorrencia'; // Verifique este caminho!

@Injectable({
  providedIn: 'root',
})
export class OcorrenciaService {
  // 2. Usar a URL do environment
  private apiUrl = environment.apiUrl;

  // Injetar o HttpClient (único construtor)
  constructor(private http: HttpClient) {}

  // Start SRT->HLS stream via backend
  startStream(options: {
    url?: string;
    streamId?: string;
    fps?: number;
    mode?: string;
    device?: string;
  }) {
    const payload: any = {
      fps: options.fps ?? 1.0,
      mode: options.mode ?? 'srt',
    };
    if (options.url) payload.url = options.url;
    if (options.streamId) payload.streamId = options.streamId;
    if (options.device) payload.device = options.device;
    return this.http.post(`${this.apiUrl}/streams/start`, payload);
  }

  /** Upload de vídeo para análise (multipart/form-data) */
  uploadAnalysis(file: File, fps?: number) {
    const fd = new FormData();
    fd.append('file', file, file.name);
    if (fps) fd.append('fps', String(fps));
    return this.http.post(`${this.apiUrl}/analysis/upload`, fd);
  }

  stopStream() {
    return this.http.post(`${this.apiUrl}/streams/stop`, {});
  }

  /**
   * Busca a lista de todas as ocorrências
   */
  getOcorrencias(): Observable<Ocorrencia[]> {
    // 4. Apontar para o endpoint correto
    return this.http.get<Ocorrencia[]>(`${this.apiUrl}/ocorrencias`);
  }

  /** Atualiza campos corrigidos pelo humano */
  updateOcorrencia(
    id: number,
    payload: {
      type?: string;
      category?: string;
      duration_s?: number;
      human_description?: string;
      severity?: string;
    }
  ): Observable<Ocorrencia> {
    return this.http.patch<Ocorrencia>(
      `${this.apiUrl}/ocorrencias/${id}`,
      payload
    );
  }

  /** Exporta CSV do backend */
  exportCsv(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/ocorrencias/export`, {
      responseType: 'blob' as 'json',
    }) as unknown as Observable<Blob>;
  }

  /** Get disk usage for a given path (GB) */
  getDiskUsage(path: string) {
    return this.http.get(`${this.apiUrl}/admin/disk-usage`, { params: { path } });
  }
}
