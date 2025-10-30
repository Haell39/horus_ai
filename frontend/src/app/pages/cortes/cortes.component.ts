// frontend/src/app/pages/cortes/cortes.component.ts
// (Versão Completa e Integrada)

import { Component, HostListener, OnInit } from '@angular/core'; // Adicionado OnInit
import { CommonModule, DatePipe } from '@angular/common'; // Adicionado DatePipe
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';

// === NOVOS IMPORTS PARA API ===
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia'; // Nosso modelo da API
import { environment } from '../../../environments/environment';

// Interface Falha (Mantida, é o nosso "ViewModel")
interface Falha {
  titulo: string;
  descricao: string;
  data: string;
  dataCompleta: string;
  horario: string;
  duracao: string;
  videoUrl: string;
  icone: string;
  tipo: string;
  dataISO: string;
  id: number;
  raw: Ocorrencia; // referência ao objeto original
}

@Component({
  selector: 'app-cortes',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  providers: [DatePipe], // === ADICIONADO DatePipe ===
  templateUrl: './cortes.component.html',
  styleUrls: ['./cortes.component.css'],
})
export class CortesComponent implements OnInit {
  // === IMPLEMENTADO OnInit ===
  filtroTexto = '';
  filtroTipo = '';
  filtroData = '';
  falhaSelecionada: Falha | null = null;
  // edição humana
  edicaoDescricao: string = '';
  edicaoTipo: string = '';
  editMode = false;
  edicaoCategoria: string = '';
  edicaoDuracao: string = '';

  // exportação
  formatoExport: 'csv' | 'pdf' = 'csv';

  itensPorPagina = 10;
  paginaAtual = 1;
  mostrarDropdown = false;

  tiposFalha: string[] = []; // === AGORA É DINÂMICO ===

  // === DADOS MESTRES (vindos da API) ===
  falhasMaster: Falha[] = [];

  // === ESTADO DA UI ===
  isLoading = true;
  errorMsg: string | null = null;

  // === CONSTRUTOR ATUALIZADO ===
  constructor(
    private ocorrenciaService: OcorrenciaService,
    private datePipe: DatePipe
  ) {}

  // === FUNÇÃO DE CARGA (NOVA) ===
  ngOnInit(): void {
    this.carregarCortes();
  }

  carregarCortes(): void {
    this.isLoading = true;
    this.errorMsg = null;
    this.ocorrenciaService.getOcorrencias().subscribe(
      (data: Ocorrencia[]) => {
        console.log(`CortesComponent: Recebidos ${data.length} ocorrências.`);

        // 1. Transforma os dados da API para o formato 'Falha'
        this.falhasMaster = data.map((oc) => this.transformarOcorrencia(oc));

        // 2. Popula os filtros de tipo dinamicamente
        const tipos = this.falhasMaster.map((f) => f.tipo);
        this.tiposFalha = [...new Set(tipos)]; // Pega tipos únicos

        this.isLoading = false;
      },
      (error) => {
        console.error('Erro ao carregar cortes:', error);
        this.errorMsg = 'Falha ao carregar clipes da API.';
        this.isLoading = false;
      }
    );
  }

  // === FUNÇÃO DE TRANSFORMAÇÃO (NOVA) ===
  private transformarOcorrencia(oc: Ocorrencia): Falha {
    const dataISO = oc.start_ts ? new Date(oc.start_ts).toISOString() : '';

    // Constrói URL absoluto para o clipe real servido pelo backend (/clips/...)
    const clipPath = (oc.evidence?.['path'] as string) || (oc.evidence?.['clip_path'] as string) || '';
    let videoUrl = 'assets/videos/placeholder_clip.mp4';
    if (clipPath) {
      if (/^https?:\/\//i.test(clipPath)) {
        videoUrl = clipPath;
      } else if (clipPath.startsWith('/')) {
        videoUrl = `${environment.backendBase}${clipPath}`;
      } else {
        videoUrl = `${environment.backendBase}/clips/${clipPath}`;
      }
    }

    return {
      titulo: oc.type || 'Falha Indefinida', // Ex: "Ruído / chiado"
      descricao: this.humanizarDescricao(oc),
      data: this.datePipe.transform(oc.start_ts, 'dd/MM/yy, HH:mm') || 'N/A',
      dataCompleta: this.datePipe.transform(oc.start_ts, 'dd/MM/yyyy') || 'N/A',
      horario: this.datePipe.transform(oc.start_ts, 'HH:mm:ss') || 'N/A',
      duracao: `${(oc.duration_s || 0).toFixed(1)} segundos`,
      videoUrl: videoUrl,
      icone: this.getIcone(oc.category),
      tipo: oc.type || 'Indefinido', // Para o filtro 'tipo'
      dataISO: this.datePipe.transform(oc.start_ts, 'yyyy-MM-dd') || '', // Para o filtro 'data'
      id: oc.id,
      raw: oc,
    };
  }

  private humanizarDescricao(oc: Ocorrencia): string {
    const ev = (oc.evidence as any) || {};
    const partes: string[] = [];
    if (ev.human_description) return ev.human_description as string;
    if (oc.category) partes.push(`Categoria: ${oc.category}`);
    if (oc.type) partes.push(`Tipo: ${oc.type}`);
    if (oc.severity) partes.push(`Severidade: ${oc.severity}`);
    if (typeof oc.confidence === 'number') partes.push(`Confiança: ${(oc.confidence * 100).toFixed(1)}%`);
    if (oc.duration_s) partes.push(`Duração: ${oc.duration_s.toFixed(1)}s`);
    return partes.join(' • ');
  }

  // === FUNÇÃO HELPER DE ÍCONE (NOVA) ===
  private getIcone(category: string | undefined): string {
    if (!category) return 'bi bi-question-circle-fill';
    if (category.toLowerCase().includes('áudio')) {
      return 'bi bi-volume-up-fill';
    }
    if (category.toLowerCase().includes('vídeo')) {
      return 'bi bi-camera-video-off-fill';
    }
    if (category.toLowerCase().includes('legenda')) {
      return 'bi bi-badge-cc-fill';
    }
    return 'bi bi-exclamation-triangle-fill'; // Padrão
  }

  // === LÓGICA DE FILTRO E PAGINAÇÃO (MODIFICADA) ===
  // Agora ela opera sobre 'falhasMaster'

  /** 🔹 Filtragem + Paginação */
  get falhasFiltradas(): Falha[] {
    const filtradas = this.falhasMaster.filter((f) => {
      const textoOk =
        !this.filtroTexto ||
        f.titulo.toLowerCase().includes(this.filtroTexto.toLowerCase()) ||
        f.descricao.toLowerCase().includes(this.filtroTexto.toLowerCase());
      const tipoOk = !this.filtroTipo || f.tipo === this.filtroTipo;
      const dataOk = !this.filtroData || f.dataISO === this.filtroData;
      return textoOk && tipoOk && dataOk;
    });

    // Atualiza o total de páginas dinamicamente
    this.totalPaginas = Math.max(
      1,
      Math.ceil(filtradas.length / this.itensPorPagina)
    );

    // Corrige paginação se a página atual for maior que o total
    if (this.paginaAtual > this.totalPaginas) {
      this.paginaAtual = this.totalPaginas;
    }

    const inicio = (this.paginaAtual - 1) * this.itensPorPagina;
    const fim = inicio + this.itensPorPagina;
    return filtradas.slice(inicio, fim);
  }

  /** 🔹 Total de páginas calculado dinamicamente */
  totalPaginas = 1;

  /** 🔹 Métodos de interação (MANTIDOS) */
  getDescricaoCurta(descricao: string, tamanho: number = 50): string {
    if (!descricao) return '';
    return descricao.length > tamanho
      ? descricao.slice(0, tamanho) + '...'
      : descricao;
  }

  selecionarFalha(falha: Falha) {
    this.falhaSelecionada = this.falhaSelecionada === falha ? null : falha;
    if (this.falhaSelecionada) {
      this.edicaoDescricao = this.falhaSelecionada.raw?.evidence?.['human_description'] || this.falhaSelecionada.descricao || '';
      this.edicaoTipo = this.falhaSelecionada.titulo || '';
      this.edicaoCategoria = this.falhaSelecionada.raw?.category || '';
      this.edicaoDuracao = (this.falhaSelecionada.raw?.duration_s ?? '').toString();
      this.editMode = false;
    }
  }

  getCor(falha: Falha): string {
    // Vamos basear a cor na categoria (que mapeamos para o ícone)
    if (falha.icone.includes('volume')) return '#3498db'; // Azul (Áudio)
    if (falha.icone.includes('camera')) return '#e74c3c'; // Vermelho (Vídeo)
    if (falha.icone.includes('badge')) return '#f1c40f'; // Amarelo (Legenda)
    return '#7f8c8d'; // Cinza (Outros)
  }

  downloadClip(falha: Falha) {
    const link = document.createElement('a');
    link.href = falha.videoUrl;
    link.download = `${falha.titulo.replace(/ /g, '_')}.mp4`;
    link.click();
  }

  salvarEdicao(): void {
    if (!this.falhaSelecionada) return;
    const id = this.falhaSelecionada.id;
    const payload: any = {};
    if (this.edicaoTipo && this.edicaoTipo !== this.falhaSelecionada.titulo) payload.type = this.edicaoTipo;
    if (this.edicaoCategoria) payload.category = this.edicaoCategoria;
    const durNum = Number(this.edicaoDuracao);
    if (!Number.isNaN(durNum)) payload.duration_s = durNum;
    payload.human_description = this.edicaoDescricao || '';
    this.ocorrenciaService.updateOcorrencia(id, payload).subscribe({
      next: (ocAtualizada) => {
        // atualiza no master
        const idx = this.falhasMaster.findIndex(f => f.id === id);
        if (idx >= 0) {
          const atualizada = this.transformarOcorrencia(ocAtualizada as any);
          this.falhasMaster[idx] = atualizada;
          this.falhaSelecionada = atualizada;
          this.editMode = false;
        }
      },
      error: (e) => console.error('Erro ao salvar edição:', e),
    });
  }

  exportar(): void {
    if (this.formatoExport === 'csv') {
      this.ocorrenciaService.exportCsv().subscribe({
        next: (blob) => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'ocorrencias.csv';
          a.click();
          window.URL.revokeObjectURL(url);
        },
        error: (e) => console.error('Falha ao exportar CSV:', e),
      });
    } else {
      // Gera PDF simples via print de uma tabela montada em tempo real
      const linhas = this.falhasMaster.map(f => `${f.titulo} — ${f.descricao} — ${f.data}`);
      const w = window.open('', '_blank');
      if (!w) return;
      w.document.write('<html><head><title>Relatório de Ocorrências</title></head><body>');
      w.document.write('<h3>Relatório de Ocorrências</h3><ul>');
      for (const l of linhas) w.document.write(`<li>${l}</li>`);
      w.document.write('</ul></body></html>');
      w.document.close();
      w.focus();
      w.print();
      // o usuário pode "Imprimir para PDF" no diálogo do navegador
    }
  }

  mudarItensPorPagina(event: Event): void {
    const valor = (event.target as HTMLSelectElement).value;
    this.itensPorPagina = Number(valor);
    this.paginaAtual = 1; // reinicia a página
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
