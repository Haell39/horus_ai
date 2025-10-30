// frontend/src/app/pages/cortes/cortes.component.ts
import { Component, HostListener, OnInit } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';

// Serviços e modelo da API
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';

// Interface Falha (ViewModel)
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
  dataISO: string;
  visualizado: boolean; // ✅ Novo campo
}

@Component({
  selector: 'app-cortes',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  providers: [DatePipe],
  templateUrl: './cortes.component.html',
  styleUrls: ['./cortes.component.css'],
})
export class CortesComponent implements OnInit {
  // Filtros
  filtroTexto = '';
  filtroTipo = '';
  filtroSeveridade = '';
  filtroData = '';
  falhaSelecionada: Falha | null = null;

  // Paginação
  itensPorPagina = 10;
  paginaAtual = 1;
  totalPaginas = 1;
  mostrarDropdown = false;

  // Tipos de filtro
  tiposFalha: string[] = [];
  tiposSeveridade: string[] = ['A - Grave', 'B - Médio', 'C - Leve', 'X - Gravíssima'];

  // Dados da API
  falhasMaster: Falha[] = [];

  // UI
  isLoading = true;
  errorMsg: string | null = null;

  constructor(private ocorrenciaService: OcorrenciaService, private datePipe: DatePipe) {}

  ngOnInit(): void {
    this.carregarCortes();
  }

  carregarCortes(): void {
    this.isLoading = true;
    this.errorMsg = null;

    this.ocorrenciaService.getOcorrencias().subscribe(
      (data: Ocorrencia[]) => {
        // Transformar dados da API para Falha
        this.falhasMaster = data.map((oc) => this.transformarOcorrencia(oc));

        // Popula tipos de falha únicos
        const tipos = this.falhasMaster.map((f) => f.tipo);
        this.tiposFalha = [...new Set(tipos)];

        this.isLoading = false;
      },
      (error) => {
        console.error('Erro ao carregar cortes:', error);
        this.errorMsg = 'Falha ao carregar clipes da API.';
        this.isLoading = false;
      }
    );
  }

  private transformarOcorrencia(oc: Ocorrencia): Falha {
    return {
      titulo: oc.type || 'Falha Indefinida',
      descricao: oc.category || 'Sem categoria',
      data: this.datePipe.transform(oc.start_ts, 'dd/MM/yy, HH:mm') || 'N/A',
      dataCompleta: this.datePipe.transform(oc.start_ts, 'dd/MM/yyyy') || 'N/A',
      horario: this.datePipe.transform(oc.start_ts, 'HH:mm:ss') || 'N/A',
      programa: 'Programa N/A',
      duracao: `${(oc.duration_s || 0).toFixed(1)} segundos`,
      videoUrl: oc.evidence?.['path'] || 'assets/videos/placeholder_clip.mp4',
      icone: this.getIcone(oc.category),
      tipo: oc.type || 'Indefinido',
      severidade: oc.severity || 'B - Médio',
      dataISO: this.datePipe.transform(oc.start_ts, 'yyyy-MM-dd') || '',
      visualizado: false // ✅ Inicialmente não visualizado
    };
  }

  private getIcone(category?: string): string {
    if (!category) return 'bi bi-question-circle-fill';
    if (category.toLowerCase().includes('áudio')) return 'bi bi-volume-up-fill';
    if (category.toLowerCase().includes('vídeo')) return 'bi bi-camera-video-off-fill';
    if (category.toLowerCase().includes('legenda')) return 'bi bi-badge-cc-fill';
    return 'bi bi-exclamation-triangle-fill';
  }

  /** 🔹 Filtragem + Paginação */
  get falhasFiltradas(): Falha[] {
    const filtradas = this.falhasMaster.filter((f) => {
      const textoOk =
        !this.filtroTexto ||
        f.titulo.toLowerCase().includes(this.filtroTexto.toLowerCase()) ||
        f.descricao.toLowerCase().includes(this.filtroTexto.toLowerCase());
      const tipoOk = !this.filtroTipo || f.tipo === this.filtroTipo;
      const dataOk = !this.filtroData || f.dataISO === this.filtroData;
      const sevOk = !this.filtroSeveridade || f.severidade === this.filtroSeveridade;
      return textoOk && tipoOk && dataOk && sevOk;
    });

    this.totalPaginas = Math.max(1, Math.ceil(filtradas.length / this.itensPorPagina));
    if (this.paginaAtual > this.totalPaginas) this.paginaAtual = this.totalPaginas;

    const inicio = (this.paginaAtual - 1) * this.itensPorPagina;
    const fim = inicio + this.itensPorPagina;
    return filtradas.slice(inicio, fim);
  }

  selecionarFalha(falha: Falha) {
    this.falhaSelecionada = this.falhaSelecionada === falha ? null : falha;

    // ✅ Marca como visualizado ao clicar
    if (falha && !falha.visualizado) {
      falha.visualizado = true;
    }
  }

  getDescricaoCurta(descricao: string, tamanho: number = 50): string {
    if (!descricao) return '';
    return descricao.length > tamanho ? descricao.slice(0, tamanho) + '...' : descricao;
  }

  getCor(falha: Falha): string {
    if (falha.icone.includes('volume')) return '#3498db';
    if (falha.icone.includes('camera')) return '#e74c3c';
    if (falha.icone.includes('badge')) return '#f1c40f';
    return '#7f8c8d';
  }

  getCorSeveridade(sev: string): string {
    switch (sev) {
      case 'A - Grave': return '#e74c3c';
      case 'B - Médio': return '#f1c40f';
      case 'C - Leve': return '#2ecc71';
      case 'X - Gravíssima': return '#9b59b6';
      default: return '#7f8c8d';
    }
  }

  downloadClip(falha: Falha) {
    const link = document.createElement('a');
    link.href = falha.videoUrl;
    link.download = `${falha.titulo.replace(/ /g, '_')}.mp4`;
    link.click();
  }

  mudarItensPorPagina(event: Event) {
    this.itensPorPagina = Number((event.target as HTMLSelectElement).value);
    this.paginaAtual = 1;
  }

  paginaAnterior() { if (this.paginaAtual > 1) this.paginaAtual--; }
  proximaPagina() { if (this.paginaAtual < this.totalPaginas) this.paginaAtual++; }
  toggleDropdown() { this.mostrarDropdown = !this.mostrarDropdown; }

  @HostListener('document:click', ['$event'])
  onClickOutside(event: Event) {
    const target = event.target as HTMLElement;
    const menu = document.querySelector('.menu-itens');
    if (menu && !menu.contains(target)) this.mostrarDropdown = false;
  }
}
