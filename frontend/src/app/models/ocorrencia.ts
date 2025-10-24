// frontend/src/app/models/ocorrencia.ts (ou ocorrencia.model.ts)

export interface Ocorrencia {
  // Campos do novo schema do backend
  id: number;
  start_ts: string;     // Padrão de API é string ISO (podemos converter pra Date)
  end_ts: string;
  created_at: string;
  duration_s?: number;  // '?' significa que é opcional
  category?: string;
  type?: string;
  severity?: string;
  confidence?: number;
  evidence?: { [key: string]: any }; // Um objeto JSON genérico
}