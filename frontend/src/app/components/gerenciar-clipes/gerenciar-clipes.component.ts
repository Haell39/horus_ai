import { Component, EventEmitter, Output, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface Clip {
  titulo: string;
  data: string;
  tamanho: string;
  selecionado: boolean;
}

@Component({
  selector: 'app-gerenciar-clipes',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './gerenciar-clipes.component.html',
  styleUrls: ['./gerenciar-clipes.component.css']
})
export class GerenciarClipesComponent {

  @Output() fechar = new EventEmitter<void>();

  storageUsed = 4.2; // GB usados
  storageTotal = 10;  // GB totais

  clips: Clip[] = [
    { titulo: 'Erro Grave', data: '14/01/2024', tamanho: '145 MB', selecionado: false },
    { titulo: 'Falha Transmissão', data: '13/01/2024', tamanho: '230 MB', selecionado: false },
    { titulo: 'Perda de Sinal', data: '12/01/2024', tamanho: '187 MB', selecionado: false },
    { titulo: 'Erro Gravíssimo', data: '11/01/2024', tamanho: '312 MB', selecionado: false },
    { titulo: 'Instabilidade', data: '10/01/2024', tamanho: '156 MB', selecionado: false },
    { titulo: 'Problema Audio', data: '09/01/2024', tamanho: '180 MB', selecionado: false },
    { titulo: 'Erro Encoder', data: '08/01/2024', tamanho: '200 MB', selecionado: false },
    { titulo: 'Erro de Delay', data: '07/01/2024', tamanho: '130 MB', selecionado: false },
    { titulo: 'Falha de Buffer', data: '06/01/2024', tamanho: '178 MB', selecionado: false },
    { titulo: 'Frame Perdido', data: '05/01/2024', tamanho: '150 MB', selecionado: false },
    { titulo: 'Erro Grave 2', data: '04/01/2024', tamanho: '145 MB', selecionado: false },
    { titulo: 'Falha Sinal 2', data: '03/01/2024', tamanho: '220 MB', selecionado: false },
  ];

  todosSelecionados = false;

  // Paginação
  itensPorPagina = 10;
  paginaAtual = 1;
  totalPaginas = 1;
  mostrarDropdown = false;

  /** 🔹 Clipes filtrados + paginação */
  get clipsPaginados(): Clip[] {
    this.totalPaginas = Math.max(1, Math.ceil(this.clips.length / this.itensPorPagina));
    const inicio = (this.paginaAtual - 1) * this.itensPorPagina;
    const fim = inicio + this.itensPorPagina;
    return this.clips.slice(inicio, fim);
  }

  /** 🔹 Barra de progresso */
  get storagePercent(): number {
    return (this.storageUsed / this.storageTotal) * 100;
  }

  /** 🔹 Selecionar todos */
  toggleSelecionarTodos(): void {
    this.todosSelecionados = !this.todosSelecionados;
    this.clipsPaginados.forEach(c => c.selecionado = this.todosSelecionados);
  }

  /** 🔹 Contar selecionados */
  get selecionadosCount(): number {
    return this.clips.filter(c => c.selecionado).length;
  }

  /** 🔹 Limpar selecionados */
  limparSelecionados(): void {
    this.clips = this.clips.filter(c => !c.selecionado);
    this.todosSelecionados = false;
    this.atualizarStorage();
  }

  /** 🔹 Limpar tudo */
  limparTudo(): void {
    this.clips = [];
    this.todosSelecionados = false;
    this.atualizarStorage();
  }

  /** 🔹 Atualiza storage usado dinamicamente */
  atualizarStorage(): void {
    const usadoMB = this.clips.reduce((acc, c) => acc + parseInt(c.tamanho), 0);
    this.storageUsed = Math.min(this.storageTotal, usadoMB / 1024);
  }

  /** 🔹 Paginação */
  paginaAnterior(): void {
    if (this.paginaAtual > 1) this.paginaAtual--;
  }

  proximaPagina(): void {
    if (this.paginaAtual < this.totalPaginas) this.paginaAtual++;
  }

  mudarItensPorPagina(event: Event): void {
    const valor = (event.target as HTMLSelectElement).value;
    this.itensPorPagina = Number(valor);
    this.paginaAtual = 1;
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

  /** 🔹 Fechar modal */
  fecharModal(): void {
    this.fechar.emit();
  }
}
