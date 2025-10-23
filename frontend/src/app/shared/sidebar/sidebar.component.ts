import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { TemaService } from '../../services/preloaderService/tema.service';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.css']
})
export class SidebarComponent implements OnInit {
  situacaoMenu: string = 'aberto';
  situacaoMenuMobile: string = 'fechado';
  nomeUsuario: string = '';
  tema: 'claro' | 'escuro' = 'escuro';
  logoSrc: string = 'icone-globo.png';

  constructor(private temaService: TemaService) {}

  ngOnInit(): void {
    const estadoSalvo = localStorage.getItem('estadoMenu');
    if (estadoSalvo) {
      this.situacaoMenu = estadoSalvo;
      this.aplicarEstadoMenu();
    }

    const nome = localStorage.getItem('nomeUsuario');
    this.nomeUsuario = nome ? nome : 'Usuário';

    this.atualizarLogo(this.temaService.isModoEscuro() ? 'escuro' : 'claro');

    // 🔁 Observa mudanças de tema (dark/light)
    const observer = new MutationObserver(() => {
      const isDark = document.body.classList.contains('dark-theme');
      this.atualizarLogo(isDark ? 'escuro' : 'claro');
    });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  }

  atualizarLogo(tema: 'escuro' | 'claro'): void {
    this.tema = tema;
    const nav = document.getElementById('nav');

    if (tema === 'claro') {
      this.logoSrc = 'icone-escuro.png';
      nav?.classList.add('sidebar-clara');
    } else {
      this.logoSrc = 'icone-globo.png';
      nav?.classList.remove('sidebar-clara');
    }
  }

  aplicarEstadoMenu(): void {
    if (this.situacaoMenu === 'fechado') this.fecharMenuPC();
    else this.abrirMenuPC();
  }

  toggleMenu(): void {
    if (window.innerWidth > 800) {
      this.situacaoMenu === 'aberto' ? this.fecharMenuPC() : this.abrirMenuPC();
    } else {
      this.situacaoMenuMobile === 'fechado' ? this.abrirMenuMobile() : this.fecharMenuMobile();
    }
  }

  fecharMenuPC(): void {
    const nav = document.getElementById('nav');
    const textMenus = document.querySelectorAll('.text-menu');
    const itensMenus = document.querySelectorAll('.itens-menu');
    const containMain = document.getElementById('coteudo-geral');

    if (nav) nav.style.left = '-130px';
    textMenus.forEach(menu => (menu as HTMLElement).style.display = 'none');
    itensMenus.forEach(item => (item as HTMLElement).style.justifyContent = 'end');
    if (containMain) containMain.style.marginLeft = '3.5%';

    this.situacaoMenu = 'fechado';
    localStorage.setItem('estadoMenu', 'fechado');
  }

  abrirMenuPC(): void {
    const nav = document.getElementById('nav');
    const textMenus = document.querySelectorAll('.text-menu');
    const itensMenus = document.querySelectorAll('.itens-menu');
    const containMain = document.getElementById('coteudo-geral');

    if (nav) nav.style.left = '0';
    textMenus.forEach(menu => (menu as HTMLElement).style.display = 'block');
    itensMenus.forEach(item => {
      (item as HTMLElement).style.justifyContent = 'start';
      (item as HTMLElement).style.marginRight = '0';
    });
    if (containMain) containMain.style.marginLeft = '195px';

    this.situacaoMenu = 'aberto';
    localStorage.setItem('estadoMenu', 'aberto');
  }

  abrirMenuMobile(): void {
    const nav = document.getElementById('nav');
    if (nav) {
      nav.style.height = '100%';
      nav.style.overflow = 'overlay';
    }
    this.situacaoMenuMobile = 'aberto';
  }

  fecharMenuMobile(): void {
    const nav = document.getElementById('nav');
    if (nav) {
      nav.style.height = '65px';
      nav.style.overflow = 'hidden';
    }
    this.situacaoMenuMobile = 'fechado';
  }
}
