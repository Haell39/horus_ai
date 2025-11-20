import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';

interface AccessibilityConfig {
  highContrast: boolean;
  reduceMotion: boolean;
  captionsEnabled: boolean;
  fontSize: number;
  colorPalette: string;
}

@Component({
  selector: 'app-acessibilidade',
  standalone: true,
  imports: [CommonModule, FormsModule, SidebarComponent],
  templateUrl: './acessibilidade.component.html',
  styleUrls: ['./acessibilidade.component.css'],
})
export class AcessibilidadeComponent implements OnInit {
  config: AccessibilityConfig = {
    highContrast: false,
    reduceMotion: true,
    captionsEnabled: true,
    fontSize: 16,
    colorPalette: 'default',
  };

  mockNotifications = [
    { id: 1, text: 'Alerta de volume alto', severity: 'grave' },
    { id: 2, text: 'Perda de sincronia detectada', severity: 'simples' },
  ];

  ngOnInit(): void {
    const saved = localStorage.getItem('acessibilidade_config');
    if (saved) {
      try {
        this.config = JSON.parse(saved);
      } catch (e) {
        // ignore
      }
    }
  }

  save(): void {
    localStorage.setItem('acessibilidade_config', JSON.stringify(this.config));
    alert('Preferências de acessibilidade salvas localmente.');
  }

  runScreenReaderTest(): void {
    // Simple mock: speak the first notification using Web Speech API if available
    try {
      const anyWin: any = window as any;
      if (anyWin.speechSynthesis) {
        const utter = new SpeechSynthesisUtterance(
          this.mockNotifications[0].text
        );
        anyWin.speechSynthesis.cancel();
        anyWin.speechSynthesis.speak(utter);
      } else {
        alert('Speech Synthesis API não disponível neste navegador');
      }
    } catch (e) {
      console.warn('Screen reader test failed', e);
    }
  }

  toggleContrast(): void {
    this.config.highContrast = !this.config.highContrast;
    document.body.classList.toggle(
      'high-contrast-mode',
      this.config.highContrast
    );
  }
}
