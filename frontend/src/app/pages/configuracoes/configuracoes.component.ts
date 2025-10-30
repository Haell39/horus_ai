import { Component, OnInit } from '@angular/core';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { TemaService } from '../../services/preloaderService/tema.service';
import { GerenciarClipesComponent } from '../../components/gerenciar-clipes/gerenciar-clipes.component';
import { CommonModule } from '@angular/common';
import { OcorrenciaService } from '../../services/ocorrencia.service';

interface Configuracoes {
  somAlerta: string;
  formatoRelatorio: string;
}

@Component({
  selector: 'app-configuracoes',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent, GerenciarClipesComponent],
  templateUrl: './configuracoes.component.html',
  styleUrls: ['./configuracoes.component.css']
})
export class ConfiguracoesComponent implements OnInit {

  configuracoes: Configuracoes = {
    somAlerta: 'beep',
    formatoRelatorio: 'padrao'
  };

  storageUsed = 4.2; // GB usados
  storageTotal = 10;  // GB totais

  mostrarModal = false; // Modal de Gerenciar Clipes

  constructor(public temaService: TemaService, private ocorrenciaService: OcorrenciaService) {}

  ngOnInit(): void {
    const saved = localStorage.getItem('configuracoes');
    if (saved) {
      this.configuracoes = JSON.parse(saved);
    }
  }

  alternarTema(): void {
    this.temaService.alternarTema();
  }

  get isModoEscuro(): boolean {
    return this.temaService.isModoEscuro();
  }

  salvarTudo(): void {
    localStorage.setItem('configuracoes', JSON.stringify(this.configuracoes));
    alert('Configurações salvas com sucesso!');
  }

  get storagePercent(): number {
    return (this.storageUsed / this.storageTotal) * 100;
  }

  // ==============================
  // TOCAR SOM DIRETAMENTE AQUI
  // ==============================
  testarSom(): void {
    const sons: { [key: string]: string[] } = {
      beep: ['Alerta-Curto.mp3'],
      digital: ['Alerta-Digital.mp3'],
      alerta: ['Alerta-Sutil.mp3']
    };

    const tipo = this.configuracoes.somAlerta;
    const caminhos = sons[tipo] || sons['beep'];
    let reproduzido = false;

    for (const caminho of caminhos) {
      const audio = new Audio(caminho);
      audio.volume = 0.8;

      audio.addEventListener('canplaythrough', () => {
        if (!reproduzido) {
          audio.play().catch(err => console.error('Erro ao tocar som:', err));
          reproduzido = true;
        }
      });

      audio.addEventListener('error', (e) => {
        console.warn(`Não foi possível carregar ${caminho}:`, e);
      });
    }

    if (caminhos.length === 0) {
      console.error(`Nenhum arquivo de áudio encontrado para o tipo "${tipo}"`);
    }
  }

  /** 🔹 Modal de Gerenciar Clipes */
  abrirGerenciarClipes(): void {
    this.mostrarModal = true;
  }

  fecharGerenciarClipes(): void {
    this.mostrarModal = false;
  }

  /** Exportação */
  exportar(): void {
    const formato = (this.configuracoes.formatoRelatorio || 'padrao').toLowerCase();
    if (formato === 'csv') {
      this.ocorrenciaService.exportCsv().subscribe({
        next: (blob) => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'ocorrencias.csv';
          a.click();
          window.URL.revokeObjectURL(url);
        },
        error: (e) => alert('Falha ao exportar CSV'),
      });
    } else if (formato === 'pdf') {
      // PDF simples: instruir o usuário a usar a exportação via Cortes ou imprimir
      const w = window.open('', '_blank');
      if (!w) return;
      w.document.write('<html><head><title>Relatório de Ocorrências</title></head><body>');
      w.document.write('<h3>Relatório de Ocorrências</h3><p>Use a página Cortes para um PDF detalhado ou selecione CSV aqui.</p>');
      w.document.write('</body></html>');
      w.document.close();
      w.focus();
      w.print();
    } else {
      alert('Selecione CSV ou PDF em Formato de Relatórios');
    }
  }
}
