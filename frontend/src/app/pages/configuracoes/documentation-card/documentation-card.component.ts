import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DocsService } from '../../../services/docs.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-documentation-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './documentation-card.component.html',
  styleUrls: ['./documentation-card.component.css'],
})
export class DocumentationCardComponent implements OnInit {
  lists: any = { docs: [], tools: [], configs: [] };
  loading = false;

  // modal state
  showModal = false;
  modalTitle = '';
  modalContent = '';
  modalHtml: SafeHtml | null = null;

  constructor(private docs: DocsService, private sanitizer: DomSanitizer) {}

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.loading = true;
    this.docs.list().subscribe({
      next: (res: any) => {
        this.lists = res || { docs: [], tools: [], configs: [] };
        this.loading = false;
      },
      error: (err: any) => {
        console.error('Falha ao listar docs', err);
        this.loading = false;
      },
    });
  }

  openFile(folder: string, name: string): void {
    this.modalTitle = `${folder} / ${name}`;
    this.modalContent = 'Carregando...';
    this.showModal = true;
    this.docs.getFile(folder, name).subscribe({
      next: (res: any) => {
        this.modalContent = res?.content || '';
        const html = this.markdownToHtml(this.modalContent || '');
        this.modalHtml = this.sanitizer.bypassSecurityTrustHtml(html);
      },
      error: (err: any) => {
        console.error('Falha ao obter arquivo', err);
        this.modalContent = `Erro ao carregar: ${
          err?.error?.detail || err?.message || 'unknown'
        }`;
        this.modalHtml = this.sanitizer.bypassSecurityTrustHtml(
          `<pre>${this.modalContent}</pre>`
        );
      },
    });
  }

  close(): void {
    this.showModal = false;
    this.modalContent = '';
    this.modalHtml = null;
  }

  // Minimal Markdown -> HTML renderer for reading mode (covers headings, code blocks, lists, links, bold/italic)
  private markdownToHtml(md: string): string {
    if (!md) return '';
    // escape HTML first
    const esc = (s: string) =>
      s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    // handle code fences
    md = md.replace(/```([\s\S]*?)```/g, (m: string, code: string) => {
      return '<pre><code>' + esc(code) + '</code></pre>';
    });
    // headings
    md = md.replace(/^###### (.*$)/gim, '<h6>$1</h6>');
    md = md.replace(/^##### (.*$)/gim, '<h5>$1</h5>');
    md = md.replace(/^#### (.*$)/gim, '<h4>$1</h4>');
    md = md.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    md = md.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    md = md.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    // bold and italic
    md = md.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    md = md.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    // inline code
    md = md.replace(/`([^`]+)`/gim, '<code>$1</code>');
    // links
    md = md.replace(
      /\[([^\]]+)\]\(([^)]+)\)/gim,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
    // unordered lists -> li
    md = md.replace(/^\s*[-\*] (.*)$/gim, '<li>$1</li>');
    // wrap consecutive li into ul
    md = md.replace(/(<li>[\s\S]*?<\/li>)/gim, (m: string) => {
      if (/^<li>/m.test(m)) {
        return '<ul>' + m.replace(/\n/g, '') + '</ul>';
      }
      return m;
    });
    // paragraphs for remaining lines
    md = md.replace(
      /^(?!<h\d|<ul|<pre|<li|<blockquote|<code)(.+)$/gim,
      '<p>$1</p>'
    );
    return md;
  }
}
