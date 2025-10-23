import { Injectable, Renderer2, RendererFactory2 } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class TemaService {
  private renderer: Renderer2;
  private darkMode = true; // ✅ inicia como true
  constructor(rendererFactory: RendererFactory2) {
    this.renderer = rendererFactory.createRenderer(null, null);

    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme === 'true') {
      this.darkMode = true;
    } else if (savedTheme === 'false') {
      this.darkMode = false;
    }

    // aplica o tema inicial
    if (this.darkMode) {
      this.renderer.addClass(document.body, 'dark-theme');
      this.renderer.removeClass(document.body, 'light-theme');
    } else {
      this.renderer.addClass(document.body, 'light-theme');
      this.renderer.removeClass(document.body, 'dark-theme');
    }
  }

  alternarTema(): void {
    this.darkMode = !this.darkMode;

    if (this.darkMode) {
      this.renderer.removeClass(document.body, 'light-theme');
      this.renderer.addClass(document.body, 'dark-theme');
    } else {
      this.renderer.removeClass(document.body, 'dark-theme');
      this.renderer.addClass(document.body, 'light-theme');
    }

    localStorage.setItem('darkMode', this.darkMode.toString());
  }

  isModoEscuro(): boolean {
    return this.darkMode;
  }
}
