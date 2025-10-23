import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';

// ✅ Aplica o tema antes do Angular iniciar
const savedTheme = localStorage.getItem('darkMode');
const isDark = savedTheme === null || savedTheme === 'true'; // padrão: escuro

if (isDark) {
  document.body.classList.add('dark-theme');
  document.body.classList.remove('light-theme');
} else {
  document.body.classList.add('light-theme');
  document.body.classList.remove('dark-theme');
}

// ✅ Agora inicia o Angular já com o tema certo
bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
