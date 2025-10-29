// frontend/src/app/services/websocket.service.ts
import { Injectable, OnDestroy } from '@angular/core';
import {
  webSocket,
  WebSocketSubject,
  WebSocketSubjectConfig,
} from 'rxjs/webSocket';
import {
  Observable,
  Subject,
  timer,
  EMPTY,
  BehaviorSubject,
  Subscription,
  of,
} from 'rxjs';
import {
  retryWhen,
  delayWhen,
  tap,
  switchMap,
  catchError,
  shareReplay,
  distinctUntilChanged,
} from 'rxjs/operators';
import { environment } from '../../environments/environment'; // Para pegar a URL base

// Define a URL do WebSocket no backend
// Nota: Usamos 'ws://' ou 'wss://' (para HTTPS)
// Removemos /api/v1 pois não definimos prefixo para websockets
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
const WS_ENDPOINT =
  WS_PROTOCOL +
  environment.apiUrl.split('//')[1].replace('/api/v1', '') +
  '/ws/ocorrencias';

@Injectable({
  providedIn: 'root',
})
export class WebsocketService implements OnDestroy {
  private socket$: WebSocketSubject<any> | undefined;
  private connectionStatusSubject = new BehaviorSubject<boolean>(false);
  private messagesSubject = new Subject<any>();
  private connectionSubscription: Subscription | undefined;

  // Observables públicos
  public messages$: Observable<any> = this.messagesSubject
    .asObservable()
    .pipe(shareReplay(1)); // Compartilha a última msg
  public isConnected$: Observable<boolean> = this.connectionStatusSubject
    .asObservable()
    .pipe(distinctUntilChanged()); // Emite só quando status muda

  constructor() {
    console.log('WebSocketService: Inicializando...');
    this.connect();
  }

  ngOnDestroy(): void {
    console.log('WebSocketService: Destruindo serviço, fechando conexão.');
    this.closeConnection(true); // Fecha ao destruir o serviço
  }

  private connect(): void {
    // Evita múltiplas conexões
    if (this.socket$ && this.connectionStatusSubject.getValue()) {
      console.log('WebSocketService: Já conectado.');
      return;
    }
    // Cancela inscrição anterior se houver
    if (this.connectionSubscription) {
      this.connectionSubscription.unsubscribe();
    }

    console.log(`WebSocketService: Tentando conectar a ${WS_ENDPOINT}...`);
    this.connectionStatusSubject.next(false); // Marca como desconectado ao tentar

    const config: WebSocketSubjectConfig<any> = {
      url: WS_ENDPOINT,
      openObserver: {
        next: () => {
          console.log('WebSocketService: Conexão WS estabelecida.');
          this.connectionStatusSubject.next(true);
        },
      },
      closeObserver: {
        next: (e: CloseEvent) => {
          // Não loga como erro se foi fechamento manual (code 1000)
          if (e.code !== 1000) {
            console.warn(
              `WebSocketService: Conexão WS fechada. Code: ${e.code}, Motivo: ${
                e.reason || 'N/A'
              }`
            );
          } else {
            console.log('WebSocketService: Conexão WS fechada normalmente.');
          }
          this.connectionStatusSubject.next(false);
          this.socket$ = undefined; // Limpa a referência
          // A lógica de retryWhen cuidará da reconexão
        },
      },
      // Serializer/Deserializer para JSON
      serializer: (msg) => JSON.stringify(msg),
      deserializer: (e) => {
        try {
          return JSON.parse(e.data);
        } catch (err) {
          console.error('WS Erro parse JSON:', err);
          return null;
        }
      },
    };

    this.socket$ = webSocket(config);

    this.connectionSubscription = this.socket$
      .pipe(
        tap({
          next: (msg) => {
            if (msg) {
              console.log('WebSocketService: Mensagem recebida:', msg);
            }
          }, // Só loga se não for null
          error: (error) =>
            console.error('WebSocketService: Erro na conexão:', error),
        }),
        // Lógica de Retentativa
        retryWhen((errors) =>
          errors.pipe(
            tap((val) =>
              console.warn(
                'WebSocketService: Erro detectado, tentando reconectar em 5s...',
                val
              )
            ),
            delayWhen((_) => timer(5000)) // Espera 5 segundos
          )
        ),
        // Filtra mensagens nulas (erro no parse JSON)
        switchMap((msg) => (msg === null ? EMPTY : of(msg))),
        catchError((err) => {
          console.error(
            'WebSocketService: Erro irrecuperável ou muitas tentativas. Parando reconexão.',
            err
          );
          this.connectionStatusSubject.next(false);
          this.socket$ = undefined;
          return EMPTY; // Para a cadeia
        })
      )
      .subscribe(
        // Envia mensagens válidas para o Subject interno
        (message) => this.messagesSubject.next(message)
      );
  }

  sendMessage(msg: any): void {
    if (this.socket$ && this.connectionStatusSubject.value) {
      console.log('WebSocketService: Enviando mensagem:', msg);
      this.socket$.next(msg);
    } else {
      console.error('WebSocketService: Não conectado. Mensagem não enviada.');
      // Opcional: Tentar reconectar antes de enviar?
      // this.connect(); // Cuidado com loops infinitos aqui
    }
  }

  closeConnection(manualClose: boolean = false): void {
    if (this.socket$) {
      console.log('WebSocketService: Fechando conexão WS.');
      if (manualClose) {
        this.socket$.complete(); // Fecha o observable e a conexão (code 1000)
      } else {
        this.socket$.unsubscribe(); // Apenas cancela a inscrição localmente
      }
      // Cancela a inscrição principal de conexão/retry
      if (this.connectionSubscription) {
        this.connectionSubscription.unsubscribe();
        this.connectionSubscription = undefined;
      }
      this.socket$ = undefined;
      this.connectionStatusSubject.next(false);
    }
  }
}
