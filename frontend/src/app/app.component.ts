import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PreloaderComponent } from '../app/shared/preloader/preloader.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,PreloaderComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'Squad_Globo';
}
