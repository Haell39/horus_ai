// preloader.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class PreloaderService {
  private loadingSubject = new BehaviorSubject<boolean>(false);
  loading$ = this.loadingSubject.asObservable();

  private preloaderColorSubject = new BehaviorSubject<string>('default');
  preloaderColor$ = this.preloaderColorSubject.asObservable();

  setLoading(state: boolean, type: string): void {
    if (type === 'requisicao') {
      this.preloaderColorSubject.next('requisicao'); 
      this.loadingSubject.next(state);
    } else {
      this.preloaderColorSubject.next('default'); 
      this.loadingSubject.next(state);
    }
  }
}