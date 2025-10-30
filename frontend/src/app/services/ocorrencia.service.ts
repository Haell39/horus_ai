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
  startStream(options: { url?: string; streamId?: string; fps?: number }) {
    const payload: any = {
      fps: options.fps ?? 1.0,
    };
    if (options.url) payload.url = options.url;
    if (options.streamId) payload.streamId = options.streamId;
    return this.http.post(`${this.apiUrl}/streams/start`, payload);
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
    payload: { type?: string; human_description?: string }
  ): Observable<Ocorrencia> {
    return this.http.patch<Ocorrencia>(`${this.apiUrl}/ocorrencias/${id}`, payload);
  }

  /** Exporta CSV do backend */
  exportCsv(): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/ocorrencias/export`, {
      responseType: 'blob' as 'json',
    }) as unknown as Observable<Blob>;
  }
}
