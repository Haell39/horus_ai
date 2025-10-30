import { Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';

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
}

@Component({
  selector: 'app-cortes',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  templateUrl: './cortes.component.html',
  styleUrls: ['./cortes.component.css'],
})
export class CortesComponent {
  filtroTexto = '';
  filtroTipo = '';
  filtroData = '';
  filtroSeveridade = '';
  falhaSelecionada: Falha | null = null;

  itensPorPagina = 10;
  paginaAtual = 1;
  mostrarDropdown = false;

  tiposFalha: string[] = ['Fade', 'Freeze'];
  tiposSeveridade: string[] = ['A - Grave', 'B - Médio', 'C - Leve', 'X - Gravíssima'];

  falhas: Falha[] = [
    {
      titulo: 'Fade na Abertura',
      descricao: 'Fade aplicado no início do vídeo.',
      data: 'Dia 23/09, 21:36',
      dataCompleta: '2025-09-23',
      horario: '21:36',
      programa: 'Primeiro Vídeo',
      duracao: '00:30',
      videoUrl: 'assets/videos/video1.mp4',
      icone: 'bi bi-play-circle-fill',
      tipo: 'Fade',
      severidade: 'B - Médio',
      dataISO: '2025-09-23',
    },
    {
      titulo: 'Freeze na Cena 2',
      descricao: 'Freeze inesperado durante a cena.',
      data: 'Dia 24/09, 14:20',
      dataCompleta: '2025-09-24',
      horario: '14:20',
      programa: 'Segundo Vídeo',
      duracao: '00:45',
      videoUrl: 'assets/videos/video2.mp4',
      icone: 'bi bi-pause-circle-fill',
      tipo: 'Freeze',
      severidade: 'A - Grave',
      dataISO: '2025-09-24',
    },

    // Adicione outras falhas conforme necessário
  ];

  /** 🔹 Filtragem + Paginação */
  get falhasFiltradas(): Falha[] {
    const filtradas = this.falhas.filter((f) => {
      const textoOk =
        !this.filtroTexto ||
        f.titulo.toLowerCase().includes(this.filtroTexto.toLowerCase()) ||
        f.descricao.toLowerCase().includes(this.filtroTexto.toLowerCase());
      const tipoOk = !this.filtroTipo || f.tipo === this.filtroTipo;
      const dataOk = !this.filtroData || f.dataISO === this.filtroData;
      const severidadeOk = !this.filtroSeveridade || f.severidade === this.filtroSeveridade;
      return textoOk && tipoOk && dataOk && severidadeOk;
    });

    this.totalPaginas = Math.max(1, Math.ceil(filtradas.length / this.itensPorPagina));
    const inicio = (this.paginaAtual - 1) * this.itensPorPagina;
    const fim = inicio + this.itensPorPagina;
    return filtradas.slice(inicio, fim);
  }

  totalPaginas = 1;

  /** 🔹 Métodos de interação */
  getDescricaoCurta(descricao: string, tamanho: number = 50): string {
    if (!descricao) return '';
    return descricao.length > tamanho ? descricao.slice(0, tamanho) + '...' : descricao;
  }

  selecionarFalha(falha: Falha) {
    this.falhaSelecionada = this.falhaSelecionada === falha ? null : falha;
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
    link.download = `${falha.titulo}.mp4`;
    link.click();
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
