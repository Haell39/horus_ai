import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';

interface AccessibilityConfig {
  // Visual
  highContrast: boolean;
  colorMode:
    | 'default'
    | 'deuteranopia'
    | 'protanopia'
    | 'tritanopia'
    | 'monochrome';
  fontSize: number;
  uiScale: number;

  // Auditory
  screenReaderEnabled: boolean;
  audioDescription: boolean;
  alertVolume: number;

  // Cognitive & Motion
  reduceMotion: boolean;
  focusMode: boolean;
  autoPlayMedia: boolean;
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
    colorMode: 'default',
    fontSize: 100, // percentage
    uiScale: 100,

    screenReaderEnabled: false,
    audioDescription: false,
    alertVolume: 80,

    reduceMotion: true,
    focusMode: false,
    autoPlayMedia: false,
  };

  // Mock data for UI
  complianceScore = 98;
  systemStatus = 'Em Conformidade (WCAG 2.1 AAA)';

  shortcuts = [
    { key: 'Space', action: 'Play / Pause' },
    { key: 'M', action: 'Mute Audio' },
    { key: 'F', action: 'Fullscreen' },
    { key: 'Alt + S', action: 'Start Stream' },
    { key: 'Alt + C', action: 'Capture Mode' },
    { key: 'Esc', action: 'Close Modals' },
  ];

  ngOnInit(): void {
    const saved = localStorage.getItem('acessibilidade_config_v2');
    if (saved) {
      try {
        this.config = { ...this.config, ...JSON.parse(saved) };
      } catch (e) {
        // ignore
      }
    }
  }

  save(): void {
    localStorage.setItem(
      'acessibilidade_config_v2',
      JSON.stringify(this.config)
    );
    // Here we would trigger a toast or notification
    console.log('Config saved:', this.config);
  }

  toggleContrast(): void {
    // Mock implementation of visual toggle
    document.body.classList.toggle(
      'high-contrast-mode',
      this.config.highContrast
    );
  }

  resetDefaults(): void {
    this.config = {
      highContrast: false,
      colorMode: 'default',
      fontSize: 100,
      uiScale: 100,
      screenReaderEnabled: false,
      audioDescription: false,
      alertVolume: 80,
      reduceMotion: true,
      focusMode: false,
      autoPlayMedia: false,
    };
    this.save();
  }
}
