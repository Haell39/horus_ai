import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class DocsService {
  constructor(private http: HttpClient) {}

  list(): Observable<any> {
    return this.http.get('http://localhost:8000/api/v1/docs/list');
  }

  getFile(folder: string, name: string): Observable<any> {
    return this.http.get('http://localhost:8000/api/v1/docs/file', {
      params: { folder: folder, name: name },
    });
  }
}
