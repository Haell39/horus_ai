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

  // 3. Injetar o HttpClient (já estava correto)
  constructor(private http: HttpClient) {}

  /**
   * Busca a lista de todas as ocorrências
   */
  getOcorrencias(): Observable<Ocorrencia[]> {
    // 4. Apontar para o endpoint correto
    return this.http.get<Ocorrencia[]>(`${this.apiUrl}/ocorrencias`);
  }
}
