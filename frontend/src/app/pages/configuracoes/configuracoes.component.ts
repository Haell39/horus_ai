import { Component, OnInit } from '@angular/core';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import { TemaService } from '../../services/preloaderService/tema.service';
import { CommonModule } from '@angular/common';
import { DocumentationCardComponent } from './documentation-card/documentation-card.component';
import { OcorrenciaService } from '../../services/ocorrencia.service';

interface Configuracoes {
  somAlerta: string;
  formatoRelatorio: string;
  storageMode?: string; // 'local' | 'onedrive'
  localPath?: string;
  oneDriveLink?: string;
}

@Component({
  selector: 'app-configuracoes',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    SidebarComponent,
    DocumentationCardComponent,
  ],
  templateUrl: './configuracoes.component.html',
  styleUrls: ['./configuracoes.component.css'],
})
export class ConfiguracoesComponent implements OnInit {
  configuracoes: Configuracoes = {
    somAlerta: 'beep',
    formatoRelatorio: 'json',
    storageMode: 'local',
    localPath: '',
    oneDriveLink: '',
  };

  storageUsed = 4.2; // GB usados
  storageTotal = 10; // GB totais
  // details filled after probe
  detectedPath: string | null = null;
  oneDriveFolder: string | null = null;

  constructor(
    public temaService: TemaService,
    private ocorrenciaService: OcorrenciaService
  ) {}

  ngOnInit(): void {
    const saved = localStorage.getItem('configuracoes');
    if (saved) {
      this.configuracoes = JSON.parse(saved);
    }
    // load oneDrive link/folder if configured by ops
    try {
      const od =
        localStorage.getItem('oneDrive_link') ||
        this.configuracoes.oneDriveLink;
      this.oneDriveFolder = od || null;
      // if localPath configured, try to probe disk usage
      if (
        this.configuracoes.storageMode === 'local' &&
        this.configuracoes.localPath
      ) {
        this.checkDiskUsage(this.configuracoes.localPath);
      }
    } catch (e) {
      this.oneDriveFolder = null;
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
    // persist OneDrive link explicitly for backward compatibility
    try {
      if (this.configuracoes.oneDriveLink) {
        localStorage.setItem('oneDrive_link', this.configuracoes.oneDriveLink);
      }
    } catch (e) {}
    // feedback mínimo
    try {
      // small alert for admin in config panel
      alert('Configurações salvas com sucesso!');
    } catch (e) {}
    // Também tentamos persistir a configuração no backend (admin endpoint)
    try {
      const payload: any = {
        mode: this.configuracoes.storageMode || 'local',
        local_path: this.configuracoes.localPath || undefined,
        oneDriveLink: this.configuracoes.oneDriveLink || undefined,
      };
      this.ocorrenciaService.setStorageConfig(payload).subscribe({
        next: (res) => {
          console.log('Storage config persisted on server:', res);
        },
        error: (err) => {
          console.warn('Falha ao persistir storage config no servidor:', err);
        },
      });
    } catch (e) {
      console.warn('Erro ao chamar setStorageConfig:', e);
    }
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
      alerta: ['Alerta-Sutil.mp3'],
    };

    const tipo = this.configuracoes.somAlerta;
    const caminhos = sons[tipo] || sons['beep'];
    let reproduzido = false;

    for (const caminho of caminhos) {
      const audio = new Audio(caminho);
      audio.volume = 0.8;

      audio.addEventListener('canplaythrough', () => {
        if (!reproduzido) {
          audio.play().catch((err) => console.error('Erro ao tocar som:', err));
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

  // Gerenciar Clipes UI temporarily removed

  /** Exportação */
  exportar(): void {
    const formato = (
      this.configuracoes.formatoRelatorio || 'padrao'
    ).toLowerCase();
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
    } else if (formato === 'json') {
      this.ocorrenciaService.getOcorrencias().subscribe({
        next: (data) => {
          try {
            const blob = new Blob([JSON.stringify(data || [], null, 2)], {
              type: 'application/json',
            });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ocorrencias.json';
            a.click();
            window.URL.revokeObjectURL(url);
          } catch (e) {
            console.error('Erro ao exportar JSON', e);
            alert('Falha ao exportar JSON');
          }
        },
        error: (e) => alert('Falha ao exportar JSON'),
      });
    } else if (formato === 'pdf') {
      // Client-side PDF using jsPDF + autotable
      this.ocorrenciaService.getOcorrencias().subscribe({
        next: (data) => {
          try {
            const doc = new jsPDF('p', 'mm', 'a4');
            const title = 'Relatório de Ocorrências';
            doc.setFontSize(14);
            doc.text(title, 14, 16);

            const columns = [
              'ID',
              'Tipo',
              'Categoria',
              'Severidade',
              'Início',
              'Duração',
              'Descrição',
            ];

            const rows = (data || []).map((oc: any) => {
              const start = oc.start_ts
                ? new Date(oc.start_ts).toLocaleString()
                : '';
              const duration = oc.duration_s != null ? `${oc.duration_s}s` : '';
              const desc =
                (oc.evidence &&
                  (oc.evidence.human_description ||
                    oc.evidence.description ||
                    '')) ||
                '';
              return [
                oc.id,
                oc.type || '',
                oc.category || '',
                oc.severity || '',
                start,
                duration,
                desc,
              ];
            });

            autoTable(doc as any, {
              startY: 22,
              head: [columns],
              body: rows,
              styles: { fontSize: 9, cellPadding: 3 },
              headStyles: { fillColor: [30, 30, 30], textColor: 255 },
              alternateRowStyles: { fillColor: [245, 245, 245] },
              margin: { left: 14, right: 14 },
              didDrawPage: (dataArg: any) => {
                const pageStr =
                  'Página ' + (doc as any).internal.getNumberOfPages();
                doc.setFontSize(9);
                doc.text(
                  pageStr,
                  (doc as any).internal.pageSize.getWidth() - 30,
                  (doc as any).internal.pageSize.getHeight() - 10
                );
              },
            });

            doc.save('ocorrencias.pdf');
          } catch (e) {
            console.error('Erro gerando PDF', e);
            alert('Falha ao gerar PDF. Veja console para detalhes.');
          }
        },
        error: (e) => {
          console.error('Falha ao buscar ocorrências para PDF', e);
          alert('Falha ao gerar PDF');
        },
      });
    } else {
      alert('Selecione CSV, JSON ou PDF em Formato de Relatórios');
    }
  }

  copiarOneDriveLink(): void {
    try {
      const link = this.configuracoes.oneDriveLink || this.oneDriveFolder;
      if (!link) {
        alert('OneDrive não configurado. Informe o link na configuração.');
        return;
      }
      // abre em nova aba
      window.open(link, '_blank');
    } catch (e) {
      console.warn('Erro ao abrir OneDrive:', e);
    }
  }

  // Tenta abrir um seletor de pasta no navegador (File System Access API)
  async selectLocalFolder(): Promise<void> {
    try {
      // feature detect
      const anyWin: any = window as any;
      if (anyWin.showDirectoryPicker) {
        const handle = await anyWin.showDirectoryPicker();
        // O handle.name normalmente é o nome da pasta; navegadores NÃO expõem
        // o caminho absoluto por motivos de segurança.
        const folderName = handle.name || '';
        // Atualiza o campo com o nome (pode não ser um caminho absoluto)
        this.configuracoes.localPath = folderName;
        // Se o nome parecer um caminho absoluto (Windows 'C:\' ou '/'), tenta verificar
        if (folderName.includes(':') || folderName.startsWith('/')) {
          this.checkDiskUsage(folderName);
        } else {
          // mostra instrução para o usuário colar o caminho completo, se necessário
          alert(
            'Seleção concluída. ATENÇÃO: o navegador pode fornecer apenas o nome da pasta (ex: "Documents"), não o caminho absoluto. Cole o caminho completo (ex: D:\\Documents) no campo antes de clicar em Verificar Disco, ou execute o helper PowerShell tools\\set-storage-config.ps1 para aplicar o caminho no servidor.'
          );
        }
        return;
      }

      // Fallback: cria um input file com webkitdirectory para permitir selecionar uma pasta
      const input = document.createElement('input');
      input.type = 'file';
      (input as any).webkitdirectory = true;
      input.style.display = 'none';
      document.body.appendChild(input);
      input.addEventListener('change', (ev: any) => {
        try {
          const files: FileList = input.files as FileList;
          if (files && files.length > 0) {
            // inferir nome raiz a partir do primeiro arquivo
            const first = files[0] as any;
            // webkitRelativePath existe quando webkitdirectory é usado
            const rel = (first as any).webkitRelativePath || '';
            const parts = rel.split('/');
            const root = parts.length > 1 ? parts[0] : first.name;
            // Atualiza o campo (não é caminho absoluto)
            this.configuracoes.localPath = root;
            alert(
              'Pasta selecionada: ' +
                root +
                "\nNota: o navegador pode não fornecer o caminho completo. Cole o caminho absoluto (ex: D:\\Documents) no campo antes de clicar em Verificar Disco. Ou rode tools\\set-storage-config.ps1 para gravar o caminho diretamente no backend.'"
            );
          }
        } finally {
          document.body.removeChild(input);
        }
      });
      input.click();
    } catch (e) {
      console.warn('selectLocalFolder failed:', e);
      alert(
        'Não foi possível abrir o seletor de pasta no navegador. Cole o caminho manualmente no campo de Caminho Local.'
      );
    }
  }

  // Verifica uso do disco para um caminho local usando o backend
  checkDiskUsage(path?: string): void {
    const p = path || this.configuracoes.localPath || '';
    if (!p) {
      alert('Informe um caminho local para verificar. Ex: C:\\ ou /mnt/data');
      return;
    }
    this.ocorrenciaService.getDiskUsage(p).subscribe({
      next: (res: any) => {
        this.detectedPath = res.path || p;
        this.storageTotal = res.total_gb || 0;
        this.storageUsed = res.used_gb || 0;
        // update oneDriveFolder for UI if relevant
      },
      error: (err) => {
        console.error('Falha ao verificar disco:', err);
        alert(
          'Falha ao verificar disco: ' +
            (err?.error?.detail || err?.message || 'erro')
        );
      },
    });
  }
}
