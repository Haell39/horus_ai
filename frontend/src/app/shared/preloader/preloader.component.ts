import { Component, Renderer2 } from '@angular/core';
import { Router, NavigationStart, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { PreloaderService } from '../../../app/services/preloaderService/preloader.service';
import { Observable, fromEvent } from 'rxjs';

@Component({
  selector: 'app-preloader',
  templateUrl: './preloader.component.html',
  styleUrls: ['./preloader.component.css'],
  standalone: true,
  imports: [CommonModule],
})
export class PreloaderComponent {
  loading$: Observable<boolean>;
  preloaderColor$: Observable<string>;

  constructor(
    private router: Router,
    private preloaderService: PreloaderService,
    private renderer: Renderer2
  ) {
    this.loading$ = this.preloaderService.loading$;
    this.preloaderColor$ = this.preloaderService.preloaderColor$;

    // Observa eventos de navegação
    this.router.events.subscribe(event => {
      if (event instanceof NavigationStart) {
        if (event.url.includes('login') || event.url.includes('cadastro')) {
          this.preloaderService.setLoading(false, 'default');
        } else {
          this.preloaderService.setLoading(true, 'default');
        }
      }

      if (event instanceof NavigationEnd) {
        setTimeout(() => {
          this.preloaderService.setLoading(false, 'default');
        }, 1000);
      }
    });

    // Atualiza cores do preloader ao iniciar
    this.atualizarTemaPreloader();

    // Observa mudanças de classe no body (modo claro/escuro)
    const observer = new MutationObserver(() => {
      this.atualizarTemaPreloader();
    });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  }

  private atualizarTemaPreloader() {
    const isLight = document.body.classList.contains('light-theme');
    const preloaderBg = isLight ? '#ffffff' : '#000000';
    const spinnerColor = isLight ? '#000000' : '#ffffff';

    this.renderer.setStyle(document.documentElement, '--preloader-bg', preloaderBg);
    this.renderer.setStyle(document.documentElement, '--preloader-spinner', spinnerColor);
  }
}
