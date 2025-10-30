import { Component, HostListener, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';
import { environment } from '../../../environments/environment';

interface Falha {
  titulo: string;
  descricao: string;
  data: string;
  dataCompleta: string;
  horario: string;
  programa: string;
  duracao: string;
  videoUrl: string;
  icone: string;
  tipo: string;
  severidade: string;
  categoria?: string;
  dataISO: string;
}

@Component({
  selector: 'app-cortes',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  templateUrl: './cortes.component.html',
  styleUrls: ['./cortes.component.css'],
})
export class CortesComponent implements OnInit {
  filtroTexto = '';
  filtroTipo = '';
  filtroData = '';
  filtroSeveridade = '';
  falhaSelecionada: Falha | null = null;

  itensPorPagina = 10;
  paginaAtual = 1;
  mostrarDropdown = false;
  // Lista completa de tipos solicitada pelo usuário
  tiposFalha: string[] = [
    'Ausência áudio',
    'Volume baixo',
    'Eco',
    'Ruido/chiado',
    'Sinal de teste 1khz',
    'Freeze',
    'Fade',
    'Efeito bloco/variação',
    'Fora de foco/imagem borrada',
  ];

  // Severidade conforme o backend envia (human readable)
  tiposSeveridade: string[] = [
    'Gravíssima (X)',
    'Grave (A)',
    'Média (B)',
    'Leve (C)',
  ];

  // Categorias possíveis
  tiposCategoria: string[] = ['Áudio Técnico', 'Vídeo Técnico'];

  // filtro de categoria
  filtroCategoria = '';

  // Lista real que virá do backend (mapeada para a interface Falha usada pela UI)
  falhas: Falha[] = [];

  // Estado de edição
  editMode = false;
  editForm: {
    category?: string;
    type?: string;
    duration_s?: number | null;
    human_description?: string | null;
    severity?: string | null;
  } = {};

  /** 🔹 Filtragem + Paginação */
  get falhasFiltradas(): Falha[] {
    const normalize = (s: string) =>
      (s || '')
        .toString()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/\//g, ' ')
        .toLowerCase()
        .trim();

    const matchesFilter = (
      value: string | undefined,
      filter: string | undefined
    ) => {
      if (!filter) return true;
      if (!value) return false;
      const nVal = normalize(value);
      const nFil = normalize(filter);
      return nVal.includes(nFil) || nFil.includes(nVal);
    };

    const textoOk = (f: Falha) => {
      if (!this.filtroTexto) return true;
      const q = normalize(this.filtroTexto);
      return (
        normalize(f.titulo).includes(q) || normalize(f.descricao).includes(q)
      );
    };

    const filtradas = this.falhas.filter((f) => {
      const okTexto = textoOk(f);
      const okTipo = matchesFilter(f.tipo, this.filtroTipo);
      const okCategoria = matchesFilter(
        f.categoria || '',
        this.filtroCategoria
      );
      const okData = !this.filtroData || f.dataISO === this.filtroData;
      const okSev =
        !this.filtroSeveridade ||
        normalize(f.severidade) === normalize(this.filtroSeveridade);
      return okTexto && okTipo && okCategoria && okData && okSev;
    });

    this.totalPaginas = Math.max(
      1,
      Math.ceil(filtradas.length / this.itensPorPagina)
    );
    const inicio = (this.paginaAtual - 1) * this.itensPorPagina;
    const fim = inicio + this.itensPorPagina;
    return filtradas.slice(inicio, fim);
  }

  totalPaginas = 1;

  /** 🔹 Métodos de interação */
  getDescricaoCurta(descricao: string, tamanho: number = 50): string {
    if (!descricao) return '';
    return descricao.length > tamanho
      ? descricao.slice(0, tamanho) + '...'
      : descricao;
  }

  selecionarFalha(falha: Falha) {
    this.falhaSelecionada = this.falhaSelecionada === falha ? null : falha;
    if (this.falhaSelecionada) {
      // popula o formulário de edição com os valores atuais
      this.editForm = {
        category:
          this.falhaSelecionada.categoria ||
          this.falhaSelecionada.tipo ||
          this.falhaSelecionada.titulo,
        type: this.falhaSelecionada.tipo,
        duration_s: this.parseDurationSeconds(this.falhaSelecionada.duracao),
        human_description: this.falhaSelecionada.descricao,
        severity: this.falhaSelecionada.severidade || null,
      };
      this.editMode = false;
    } else {
      this.editMode = false;
    }
  }

  getCor(falha: Falha): string {
    switch (falha.tipo) {
      case 'Fade':
        return '#3498db';
      case 'Freeze':
        return '#e74c3c';
      default:
        return '#7f8c8d';
    }
  }

  getCorSeveridade(sev: string): string {
    if (!sev) return '#7f8c8d';
    const s = sev.toLowerCase();
    // Gravíssima (X) -> vermelho
    if (
      s.includes('gravíssima') ||
      s.includes('(x)') ||
      s.includes('gravíssima (x)')
    )
      return '#e74c3c';
    // Grave (A) -> laranja
    if (s.includes('grave') || s.includes('(a)')) return '#d35400';
    // Média (B) -> amarelo
    if (s.includes('média') || s.includes('(b)')) return '#f1c40f';
    // Leve (C) -> cinza
    if (s.includes('leve') || s.includes('(c)')) return '#95a5a6';
    return '#7f8c8d';
  }

  downloadClip(falha: Falha) {
    const link = document.createElement('a');
    link.href = falha.videoUrl;
    link.target = '_blank';
    link.download = `${falha.titulo}.mp4`;
    link.click();
  }

  /** Converte strings "00:30" ou "30" para segundos (aprox) */
  parseDurationSeconds(dur: string | undefined): number | null {
    if (!dur) return null;
    // tenta extrair números
    const parts = dur.split(':').map((s) => s.trim());
    if (parts.length === 2) {
      const mm = Number(parts[0] || 0);
      const ss = Number(parts[1] || 0);
      return mm * 60 + ss;
    }
    const n = Number(dur.replace(/[^0-9.]/g, ''));
    return isNaN(n) ? null : n;
  }

  /** Monta a URL pública do clipe a partir do campo evidence do backend */
  buildClipUrl(evidence: { [k: string]: any } | undefined): string {
    const backendBase =
      environment.backendBase || environment.apiUrl || 'http://localhost:8000';
    if (!evidence) return 'assets/videos/video1.mp4';
    const candidates = ['clip_path', 'path', 'file', 'frame'];
    for (const c of candidates) {
      const v = evidence[c];
      if (!v) continue;
      // se já for URL absoluta
      if (
        typeof v === 'string' &&
        (v.startsWith('http://') || v.startsWith('https://'))
      )
        return v;
      // se contiver '/clips/' extrai o nome
      if (typeof v === 'string' && v.includes('/clips/')) {
        const idx = v.indexOf('/clips/');
        return `${backendBase}${v.substring(idx)}`;
      }
      // se for somente o nome do arquivo
      if (typeof v === 'string') return `${backendBase}/clips/${v}`;
    }
    return 'assets/videos/video1.mp4';
  }

  // Carrega ocorrências reais do backend
  constructor(private ocorrenciaService: OcorrenciaService) {}

  ngOnInit(): void {
    this.loadOcorrencias();
  }

  loadOcorrencias() {
    this.ocorrenciaService.getOcorrencias().subscribe({
      next: (list: Ocorrencia[]) => {
        this.falhas = list.map((oc) => this.mapOcorrenciaToFalha(oc));
        this.totalPaginas = Math.max(
          1,
          Math.ceil(this.falhas.length / this.itensPorPagina)
        );
      },
      error: (err) => {
        console.error('Erro ao carregar ocorrências', err);
      },
    });
  }

  private mapOcorrenciaToFalha(oc: Ocorrencia): Falha {
    const start = new Date(oc.start_ts);
    const end = new Date(oc.end_ts);
    const data = start.toLocaleDateString();
    const horario = start.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
    const dur = oc.duration_s ? this.formatDuration(oc.duration_s) : '';
    const evidence = oc.evidence || {};
    // inferir categoria quando não estiver preenchida no banco
    let inferredCategory = oc.category || '';
    if (!inferredCategory && oc.type) {
      const t = oc.type.toLowerCase();
      const audioTypes = [
        'ausência áudio',
        'volume baixo',
        'eco',
        'ruido/chiado',
        'sinal de teste 1khz',
      ];
      const videoTypes = [
        'freeze',
        'fade',
        'efeito bloco/variação',
        'fora de foco/imagem borrada',
        'borrado',
        'bloco',
      ];
      if (audioTypes.find((a) => t.includes(a)))
        inferredCategory = 'Áudio Técnico';
      else if (videoTypes.find((v) => t.includes(v)))
        inferredCategory = 'Vídeo Técnico';
    }

    const titulo = `${oc.type || inferredCategory || 'Ocorrência'} #${oc.id}`;
    const ev = oc.evidence || {};
    const descricao = ev['human_description'] || '';
    return {
      titulo,
      descricao,
      data: `Dia ${data}, ${horario}`,
      dataCompleta: start.toISOString().slice(0, 10),
      horario,
      programa: evidence['program'] || evidence['programa'] || '',
      duracao: dur,
      videoUrl: this.buildClipUrl(evidence),
      icone:
        oc.type && oc.type.toLowerCase().includes('freeze')
          ? 'bi bi-pause-circle-fill'
          : 'bi bi-play-circle-fill',
      tipo: oc.type || oc.category || '',
      categoria: inferredCategory || oc.category || '',
      severidade: oc.severity || '',
      dataISO: start.toISOString().slice(0, 10),
    };
  }

  private humanizeOcorrencia(oc: Ocorrencia): string {
    const sb: string[] = [];
    sb.push(`ID ${oc.id}`);
    if (oc.type) sb.push(`Tipo: ${oc.type}`);
    if (oc.category) sb.push(`Categoria: ${oc.category}`);
    if (oc.severity) sb.push(`Severidade: ${oc.severity}`);
    if (oc.duration_s != null)
      sb.push(`Duração: ${this.formatDuration(oc.duration_s)}`);
    if (oc.confidence != null)
      sb.push(`Confiança: ${(oc.confidence * 100).toFixed(1)}%`);
    const ev = oc.evidence || {};
    if (ev['human_description'])
      sb.push(`Observação: ${ev['human_description']}`);
    return sb.join(' • ');
  }

  private formatDuration(sec: number): string {
    if (!isFinite(sec)) return '';
    const s = Math.round(sec);
    if (s >= 60) {
      const m = Math.floor(s / 60);
      const r = s % 60;
      return `${m.toString().padStart(2, '0')}:${r
        .toString()
        .padStart(2, '0')}`;
    }
    return `00:${s.toString().padStart(2, '0')}`;
  }

  /** Alterna para o modo de edição no painel de detalhes */
  enableEdit() {
    if (!this.falhaSelecionada) return;
    this.editMode = true;
  }

  cancelEdit() {
    this.editMode = false;
    if (this.falhaSelecionada) {
      // restaura descrição do banco (já está no objeto local se carregado)
      this.editForm.human_description = this.falhaSelecionada.descricao;
    }
  }

  saveEdit() {
    if (!this.falhaSelecionada) return;
    const idPart = (this.falhaSelecionada.titulo || '').split('#').pop();
    const id = idPart ? Number(idPart) : null;
    if (!id) return;
    const payload: any = {};
    if (this.editForm.type !== undefined) payload.type = this.editForm.type;
    if (this.editForm.category !== undefined)
      payload.category = this.editForm.category;
    if (
      this.editForm.duration_s !== undefined &&
      this.editForm.duration_s !== null
    )
      payload.duration_s = this.editForm.duration_s;
    if (this.editForm.human_description !== undefined)
      payload.human_description = this.editForm.human_description;
    if (
      this.editForm.severity !== undefined &&
      this.editForm.severity !== null &&
      this.editForm.severity !== ''
    )
      payload.severity = this.editForm.severity;

    this.ocorrenciaService.updateOcorrencia(id, payload).subscribe({
      next: (updated) => {
        // atualiza a entrada local e sai do modo edição
        this.falhas = this.falhas.map((f) =>
          f.titulo.endsWith(`#${id}`) ? this.mapOcorrenciaToFalha(updated) : f
        );
        this.falhaSelecionada =
          this.falhas.find((f) => f.titulo.endsWith(`#${id}`)) || null;
        this.editMode = false;
      },
      error: (err) => {
        console.error('Erro ao salvar edição', err);
      },
    });
  }

  mudarItensPorPagina(event: Event): void {
    const valor = (event.target as HTMLSelectElement).value;
    this.itensPorPagina = Number(valor);
    this.paginaAtual = 1;
  }

  paginaAnterior(): void {
    if (this.paginaAtual > 1) this.paginaAtual--;
  }

  proximaPagina(): void {
    if (this.paginaAtual < this.totalPaginas) this.paginaAtual++;
  }

  toggleDropdown() {
    this.mostrarDropdown = !this.mostrarDropdown;
  }

  @HostListener('document:click', ['$event'])
  onClickOutside(event: Event) {
    const target = event.target as HTMLElement;
    const menu = document.querySelector('.menu-itens');
    if (menu && !menu.contains(target)) this.mostrarDropdown = false;
  }
}
