// frontend/src/app/pages/monitoramento/monitoramento.component.ts
// (Versão Final com WebSocket Integrado)

import {
  Component,
  ViewChild,
  ElementRef,
  OnInit,
  OnDestroy,
  ChangeDetectionStrategy, // Importa ChangeDetectionStrategy
  ChangeDetectorRef, // Importa ChangeDetectorRef
} from '@angular/core';
import { Router } from '@angular/router';
import {
  ChartComponent,
  NgApexchartsModule,
  ApexAxisChartSeries,
  ApexChart,
  ApexXAxis,
  ApexYAxis,
  ApexStroke,
  ApexDataLabels,
  ApexTooltip,
  ApexLegend,
} from 'ng-apexcharts'; // Imports ApexCharts
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { CommonModule, DatePipe } from '@angular/common'; // Imports CommonModule e DatePipe
import { FormsModule } from '@angular/forms';

import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';
import { WebsocketService } from '../../services/websocket.service'; // <<< IMPORT WebSocketService
import { Subscription } from 'rxjs'; // <<< IMPORT Subscription
import { environment } from '../../../environments/environment';

export type ChartOptions = {
  series: ApexAxisChartSeries;
  chart: ApexChart;
  xaxis: ApexXAxis;
  yaxis: ApexYAxis;
  stroke: ApexStroke;
  dataLabels: ApexDataLabels;
  tooltip: ApexTooltip;
  labels: string[];
  legend: ApexLegend;
};

// Interface interna para Alertas (mantida)
interface AlertaOcorrencia {
  hora: string;
  mensagem: string;
  tipo: string;
  animacao: string;
  origem: string;
}

@Component({
  selector: 'app-monitoramento',
  standalone: true,
  imports: [SidebarComponent, NgApexchartsModule, CommonModule, FormsModule],
  providers: [DatePipe],
  templateUrl: './monitoramento.component.html',
  styleUrls: ['./monitoramento.component.css'],
  // Usa OnPush para melhor performance com atualizações via WebSocket
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MonitoramentoComponent implements OnInit, OnDestroy {
  @ViewChild('chart') chart!: ChartComponent;
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;

  // Stream mode selector: 'srt' or 'capture'
  public streamMode: 'srt' | 'capture' = 'srt';
  // optional capture device string (e.g. '/dev/video0' or dshow spec)
  public captureDevice: string = '';

  // HLS / reconnection state
  private hlsInstance: any = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 6;
  private reconnectBaseDelayMs = 1500;
  private reconnectTimer: any = null;
  public isStreaming: boolean = false; // whether backend ingest is running
  public isLive: boolean = false; // whether player has a live source attached
  // Upload analysis state
  public selectedFile: File | null = null;
  public uploadingFile: boolean = false;

  // Lista de vídeos de simulação (mantida)
  videos: Array<{
    src: string;
    tipo: string;
    descricao?: string;
    label?: string;
  }> = [
    // Não colocamos srt:// direto no src para evitar que o browser tente abrir o protocolo
    { src: '', tipo: 'L', descricao: 'Ao Vivo (Globo)', label: 'Ao Vivo' },
  ];
  videoIndex = 0;

  // Variáveis atualizadas pela API
  totalOcorrencias = 0;
  falhasGraves = 0;
  ultimaFalha: string | null = null;
  alertas: AlertaOcorrencia[] = [];

  // Variáveis de simulação do gráfico (mantidas)
  private loopInterval: any;
  public chartOptions!: ChartOptions; // Inicializado em carregarEstruturaGrafico

  // <<< VARIÁVEIS WebSocket >>>
  private wsSubscription: Subscription | null = null;
  public isWsConnected: boolean = false; // Status da conexão WS

  constructor(
    private ocorrenciaService: OcorrenciaService,
    private datePipe: DatePipe,
    private websocketService: WebsocketService, // <<< INJETADO
    private cdr: ChangeDetectorRef, // <<< INJETADO (Change Detector)
    private router: Router
  ) {}

  // Navega para a página de Cortes. opcionalmente aceita filtros no futuro.
  public goToCortes() {
    try {
      this.router.navigate(['/cortes']);
    } catch (e) {
      console.warn('Falha ao navegar para /cortes', e);
    }
  }

  ngOnInit() {
    console.log('MonitoramentoComponent: ngOnInit');
    this.carregarEstruturaGrafico(); // Inicializa chartOptions

    // Alerta inicial
    this.alertas = [
      {
        hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
        mensagem: 'Sistema iniciado.',
        tipo: 'Info (S)',
        animacao: 'aparecer',
        origem: 'Sistema',
      },
    ];

    this.carregarDadosIniciais(); // Carga HTTP
    this.startLoop(); // Loop do gráfico (simulação)
    this.conectarWebSocket(); // Conecta e escuta o WebSocket
    // Se environment.liveStreamUrl for um SRT, pede ao backend para iniciar SRT->HLS
    const live = environment.liveStreamUrl || '';
    if (live.startsWith('srt://')) {
      // solicita backend para iniciar o ingest e converte para HLS
      this.ocorrenciaService.startStream({ url: live, fps: 1.0 }).subscribe({
        next: () => {
          console.log('Solicitado backend para iniciar SRT->HLS');
          // Define o src do player para o HLS gerado pelo backend
          const hlsUrl = `${environment.backendBase}/hls/stream.m3u8`;
          const videoEl = this.videoPlayer?.nativeElement;
          if (videoEl) {
            videoEl.src = hlsUrl;
          }
          setTimeout(() => this.tryAttachHls(), 500);
        },
        error: (err) => {
          console.error('Falha ao solicitar backend iniciar stream:', err);
          try {
            const detail = err?.error?.detail || '';
            if (
              err.status === 400 &&
              detail.toString().toLowerCase().includes('already running')
            ) {
              const hlsUrl = `${environment.backendBase}/hls/stream.m3u8`;
              const videoEl = this.videoPlayer?.nativeElement;
              if (videoEl) videoEl.src = hlsUrl;
              setTimeout(() => this.tryAttachHls(), 500);
            }
          } catch (e) {
            // ignore
          }
        },
      });
    } else if (live) {
      // Se live não é SRT (por exemplo HLS), usa diretamente
      const videoEl = this.videoPlayer?.nativeElement;
      if (videoEl) {
        videoEl.src = live;
        this.isLive = true;
      }
      setTimeout(() => this.tryAttachHls(), 500);
    } else {
      // default: apenas tenta anexar HLS com fonte atual (player local)
      setTimeout(() => this.tryAttachHls(), 500);
    }
  }

  // Handler when user selects a file for analysis
  onFileSelected(event: Event) {
    try {
      const input = event.target as HTMLInputElement;
      if (!input.files || input.files.length === 0) {
        this.selectedFile = null;
        return;
      }
      this.selectedFile = input.files[0];
      this.cdr.markForCheck();
    } catch (e) {
      console.warn('Erro ao selecionar arquivo:', e);
      this.selectedFile = null;
    }
  }

  // Upload the selected file and request analysis
  uploadAnalysisClicked() {
    if (!this.selectedFile) return;
    this.uploadingFile = true;
    const fps = 1.0; // default fps for file analysis
    this.ocorrenciaService.uploadAnalysis(this.selectedFile, fps).subscribe({
      next: (res: any) => {
        console.log('Upload analysis response:', res);
        // If backend returned a created occurrence object (sync), process it like a new occurrence
        if (res && res.id) {
          this.processarNovaOcorrencia(res as any);
        } else if (res && res.status === 'ok') {
          // Mostrar notificação imediata ao usuário (alert) e também empurrar para a lista de alertas
          try {
            // Notificação simples (pode ser trocada por um toast futuro)
            window.alert(res.message || 'Arquivo analisado.');
          } catch (e) {
            // fallback silencioso
          }
          this.alertas.unshift({
            hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
            mensagem:
              res.message || 'Arquivo analisado — sem falhas detectadas.',
            tipo: 'Info (S)',
            animacao: 'aparecer',
            origem: 'Análise de Arquivo',
          });
        } else if (res && res.status === 'queued') {
          // Inform user that processing is queued
          this.alertas.unshift({
            hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
            mensagem: 'Arquivo enviado. Processamento em segundo plano.',
            tipo: 'Info (S)',
            animacao: 'aparecer',
            origem: 'Análise de Arquivo',
          });
        }
        this.selectedFile = null;
        this.uploadingFile = false;
        this.cdr.markForCheck();
      },
      error: (err) => {
        console.error('Erro ao enviar arquivo para análise:', err);
        this.alertas.unshift({
          hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
          mensagem: 'Falha ao enviar arquivo para análise',
          tipo: 'Erro (E)',
          animacao: 'aparecer',
          origem: 'Análise de Arquivo',
        });
        this.uploadingFile = false;
        this.cdr.markForCheck();
      },
    });
  }

  // Tenta carregar hls.js dinamicamente e anexar ao player (se necessário)
  private tryAttachHls(): void {
    try {
      const videoEl = this.videoPlayer?.nativeElement;
      if (!videoEl) return;
      // Preferir o src do elemento (pode ter sido setado diretamente para o HLS backend)
      const src = videoEl.src || this.videos[this.videoIndex]?.src;
      if (!src) return;
      // Se o browser suporta HLS nativo (Safari), usa e adiciona recovery
      const isNativeHls = videoEl.canPlayType('application/vnd.apple.mpegurl');
      if (isNativeHls) {
        videoEl.src = src;
        this.attachNativeRecovery(videoEl, src);
        return;
      }

      // Caso não suporte nativo, tenta carregar hls.js via CDN dinamicamente
      if (!(window as any).Hls) {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/hls.js@latest';
        script.onload = () => this.attachHlsToVideo(videoEl, src);
        script.onerror = () => {
          console.warn(
            'Falha ao carregar hls.js via CDN. Tentando fallback nativo.'
          );
          videoEl.src = src;
          this.attachNativeRecovery(videoEl, src);
        };
        document.head.appendChild(script);
      } else {
        this.attachHlsToVideo(videoEl, src);
      }
    } catch (err) {
      console.warn('tryAttachHls erro:', err);
    }
  }

  private attachHlsToVideo(videoEl: HTMLVideoElement, src: string) {
    try {
      const Hls = (window as any).Hls;
      if (!Hls) return;
      // destroy previous instance if exists
      if (this.hlsInstance && this.hlsInstance.destroy) {
        try {
          this.hlsInstance.destroy();
        } catch {}
        this.hlsInstance = null;
      }

      const hls = new Hls();
      this.hlsInstance = hls;
      hls.loadSource(src);
      hls.attachMedia(videoEl);

      // Reset reconnect attempts on successful manifest
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        this.reconnectAttempts = 0;
        this.isLive = true;
        try {
          videoEl.play().catch(() => {});
        } catch {}
      });

      // Handle errors from hls.js
      hls.on(Hls.Events.ERROR, (event: any, data: any) => {
        console.warn('hls.js error', data);
        try {
          if (data && data.type === Hls.ErrorTypes.NETWORK_ERROR) {
            console.warn('hls.js network error, tentando recuperar...');
            hls.startLoad();
            return;
          }
          if (data && data.type === Hls.ErrorTypes.MEDIA_ERROR) {
            console.warn('hls.js media error, tentando recuperar...');
            hls.recoverMediaError();
            return;
          }
        } catch (e) {
          // ignore
        }
        // fallback: schedule a full reconnect
        this.scheduleReconnect(videoEl, src);
      });
    } catch (err) {
      console.warn('attachHlsToVideo erro:', err);
    }
  }

  // Attach handlers for native HLS playback errors (Safari fallback or browser-level support)
  private attachNativeRecovery(videoEl: HTMLVideoElement, src: string) {
    const onError = () => {
      console.warn('Native HLS error detected, tentando reconectar...');
      this.scheduleReconnect(videoEl, src);
    };
    videoEl.removeEventListener('error', onError);
    videoEl.addEventListener('error', onError);
    this.isLive = true;
  }

  private scheduleReconnect(videoEl: HTMLVideoElement, src: string) {
    // If user already stopped the stream or the component is not live,
    // skip scheduling reconnects.
    if (!this.isStreaming && !this.isLive) {
      console.log(
        'scheduleReconnect: stream not live or stopped — skipping reconnect'
      );
      return;
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn('Max reconnect attempts reached. Não reconectando.');
      this.isLive = false;
      return;
    }
    this.reconnectAttempts++;
    const delay =
      this.reconnectBaseDelayMs * Math.pow(1.8, this.reconnectAttempts - 1);
    console.log(
      `Tentativa de reconexão #${this.reconnectAttempts} em ${Math.round(
        delay
      )}ms`
    );
    // clear any previous scheduled attempt
    try {
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    } catch (e) {}

    this.reconnectTimer = setTimeout(() => {
      try {
        // destroy hls instance if present
        if (this.hlsInstance && this.hlsInstance.destroy) {
          try {
            this.hlsInstance.destroy();
          } catch {}
          this.hlsInstance = null;
        }
        // reload src and reattach
        videoEl.src = src;
        setTimeout(() => this.tryAttachHls(), 300);
      } catch (e) {
        console.warn('Erro ao tentar reconectar:', e);
      }
    }, delay);
  }

  ngOnDestroy() {
    console.log('MonitoramentoComponent: ngOnDestroy');
    // Limpa inscrição WebSocket
    if (this.wsSubscription) {
      this.wsSubscription.unsubscribe();
    }
    // Limpa intervalo do gráfico
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
    // Opcional: Fechar WS se não for mais usado
    // this.websocketService.closeConnection(true);
    // destroy hls instance
    if (this.hlsInstance && this.hlsInstance.destroy) {
      try {
        this.hlsInstance.destroy();
      } catch {}
      this.hlsInstance = null;
    }
  }

  // UI handlers for Start / Stop
  public startStreamClicked() {
    const live = environment.liveStreamUrl || '';
    // Build payload depending on selected mode (srt or capture)
    const payload: any = { fps: 1.0 };
    if (this.streamMode === 'capture') {
      payload.mode = 'capture';
      // prefer explicit input, fall back to environment variable if set
      payload.device =
        this.captureDevice || (environment as any).captureDevice || '';
    } else {
      payload.mode = 'srt';
      payload.streamId = environment.streamId;
    }

    this.ocorrenciaService.startStream(payload).subscribe({
      next: () => {
        console.log('Start requested');
        // mark streaming and wait for playlist to be ready before attaching
        this.isStreaming = true;
        const videoEl = this.videoPlayer?.nativeElement;
        if (videoEl) {
          videoEl.src = `${environment.backendBase}/hls/stream.m3u8`;
          // poll playlist until it contains segments then attach
          this.waitForPlaylistThenAttach(
            `${environment.backendBase}/hls/stream.m3u8`,
            videoEl
          );
        }
      },
      error: (err) => {
        // If backend reports "Stream already running" (400), treat as success
        try {
          const detail = err?.error?.detail || '';
          if (
            err.status === 400 &&
            detail.toString().toLowerCase().includes('already running')
          ) {
            console.warn('Stream already running - attaching to existing HLS');
            this.isStreaming = true;
            const videoEl = this.videoPlayer?.nativeElement;
            if (videoEl) {
              videoEl.src = `${environment.backendBase}/hls/stream.m3u8`;
              this.waitForPlaylistThenAttach(
                `${environment.backendBase}/hls/stream.m3u8`,
                videoEl
              );
            }
            return;
          }
        } catch (e) {
          // fallthrough to generic error
        }
        console.error('Erro ao iniciar stream:', err);
      },
    });
  }

  // Poll the HLS playlist until it contains EXTINF segments, then attach HLS to the video element
  private async waitForPlaylistThenAttach(
    playlistUrl: string,
    videoEl: HTMLVideoElement
  ) {
    try {
      const maxAttempts = 20;
      for (let i = 0; i < maxAttempts; i++) {
        try {
          const res = await fetch(playlistUrl, { cache: 'no-store' });
          if (res.ok) {
            const txt = await res.text();
            if (txt && txt.includes('#EXTINF')) {
              // playlist contains segments, attach HLS now
              setTimeout(() => this.tryAttachHls(), 50);
              return;
            }
          }
        } catch (e) {
          // ignore and retry
        }
        await new Promise((r) => setTimeout(r, 250));
      }
      // fallback: try attach anyway
      setTimeout(() => this.tryAttachHls(), 300);
    } catch (e) {
      setTimeout(() => this.tryAttachHls(), 300);
    }
  }

  public stopStreamClicked() {
    this.ocorrenciaService.stopStream().subscribe({
      next: () => {
        console.log('Stop requested');
        this.isStreaming = false;
        this.isLive = false;
        // cancel any scheduled reconnect
        try {
          if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
          }
        } catch (e) {}
        this.reconnectAttempts = 0;
        const videoEl = this.videoPlayer?.nativeElement;
        if (videoEl) {
          try {
            videoEl.pause();
          } catch {}
          try {
            videoEl.removeAttribute('src');
            videoEl.load();
          } catch {}
        }
        if (this.hlsInstance && this.hlsInstance.destroy) {
          try {
            this.hlsInstance.destroy();
          } catch {}
          this.hlsInstance = null;
        }
      },
      error: (err) => {
        console.error('Erro ao parar stream:', err);
      },
    });
  }

  // Inicializa a estrutura do gráfico (essencial antes de usar this.chartOptions)
  carregarEstruturaGrafico() {
    this.chartOptions = {
      series: [{ name: 'Monitoramento', data: [0] }],
      chart: {
        type: 'line',
        height: 290,
        background: '#2b2b2b',
        foreColor: '#fff',
        animations: {
          enabled: true,
          easing: 'linear',
          dynamicAnimation: { enabled: true, speed: 1000 },
        },
        toolbar: { show: false },
      },
      dataLabels: { enabled: false },
      stroke: { curve: 'smooth', width: 4, lineCap: 'round' },
      xaxis: {
        type: 'category',
        labels: { show: false },
        axisTicks: { show: false },
        axisBorder: { show: false },
      },
      yaxis: { show: false, min: 0, max: 3 },
      labels: [],
      legend: { show: false },
      tooltip: { enabled: false },
    };
  }

  // Carrega dados históricos via HTTP
  carregarDadosIniciais() {
    console.log('MonitoramentoComponent: Buscando dados iniciais HTTP...');
    // Pega apenas os últimos 50 (ou menos) para a carga inicial
    this.ocorrenciaService.getOcorrencias().subscribe({
      // Usando objeto Observer
      next: (data: Ocorrencia[]) => {
        console.log(`HTTP: Recebidas ${data.length} ocorrências.`);
        if (data.length > 0) {
          this.totalOcorrencias = data.length; // Pode precisar buscar o total real se API paginar
          this.falhasGraves = data.filter((oc) => this.isGrave(oc)).length; // Calcula sobre todos os dados recebidos
          this.ultimaFalha = data[0].type || 'N/A'; // Assume que API retorna mais recente primeiro

          const MAX_ALERTAS_INICIAIS = 5; // Mostra os 5 mais recentes
          const alertasDaApi = data
            .slice(0, MAX_ALERTAS_INICIAIS)
            .map((oc) => this.formatarAlerta(oc)) // Formata sem animação
            .reverse(); // Inverte para mostrar mais antigo primeiro na carga inicial

          // Adiciona os históricos APÓS o alerta de "Sistema iniciado"
          this.alertas.push(...alertasDaApi);

          this.cdr.markForCheck(); // Marca para verificação com OnPush
        } else {
          console.log('HTTP: Nenhuma ocorrência inicial encontrada.');
        }
      },
      error: (error) => {
        console.error('Erro ao buscar dados iniciais:', error);
        this.alertas.push({
          // Adiciona erro APÓS o "Sistema iniciado"
          hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
          mensagem: 'Falha ao carregar histórico',
          tipo: 'Erro (E)',
          animacao: 'aparecer',
          origem: 'Sistema',
        });
        this.cdr.markForCheck(); // Marca para verificação com OnPush
      },
    });
  }

  // Conecta ao WebSocket e escuta mensagens
  conectarWebSocket(): void {
    console.log('MonitoramentoComponent: Inscrevendo-se no WebSocket...');
    // Cancela inscrição anterior, se houver
    if (this.wsSubscription) {
      this.wsSubscription.unsubscribe();
    }

    // Escuta o status
    this.websocketService.isConnected$.subscribe((status) => {
      if (this.isWsConnected !== status) {
        // Só atualiza se mudar
        this.isWsConnected = status;
        console.log(
          'MonitoramentoComponent: Status WS:',
          status ? 'Conectado' : 'Desconectado'
        );
        this.cdr.markForCheck(); // Marca para verificação
      }
    });

    // Escuta as mensagens
    this.wsSubscription = this.websocketService.messages$.subscribe({
      next: (message) => {
        console.log('MonitoramentoComponent: Mensagem WS recebida:', message);
        if (message?.type === 'nova_ocorrencia' && message.data) {
          // Converte data (que vem como string do JSON) para objeto Ocorrencia
          const novaOcorrencia = message.data as Ocorrencia;
          this.processarNovaOcorrencia(novaOcorrencia);
        } else {
          console.log(
            'MonitoramentoComponent: Mensagem WS ignorada (tipo inválido ou sem dados)'
          );
        }
      },
      error: (error) => {
        console.error('MonitoramentoComponent: Erro na subscrição WS:', error);
        this.isWsConnected = false; // Garante que status é falso em caso de erro
        this.cdr.markForCheck();
      },
    });
  }

  // Processa uma nova ocorrência recebida via WebSocket
  processarNovaOcorrencia(novaOcorrencia: Ocorrencia): void {
    console.log(
      'Processando nova ocorrência WS:',
      novaOcorrencia.id,
      novaOcorrencia.type
    );

    // ATENÇÃO: Se a API retornar o total, usar ele. Senão, incrementamos.
    // Assumindo que incrementamos por enquanto.
    this.totalOcorrencias++;
    if (this.isGrave(novaOcorrencia)) {
      this.falhasGraves++;
    }
    this.ultimaFalha = novaOcorrencia.type || 'N/A';

    const novoAlerta = this.formatarAlerta(novaOcorrencia, true); // true = com animação

    // Adiciona no INÍCIO da lista
    this.alertas.unshift(novoAlerta);

    // Limita o tamanho da lista
    const MAX_ALERTAS_NA_TELA = 10;
    if (this.alertas.length > MAX_ALERTAS_NA_TELA) {
      this.alertas.pop(); // Remove o mais antigo (do final)
    }

    // Remove a animação após um tempo
    setTimeout(() => {
      const index = this.alertas.findIndex((a) => a === novoAlerta);
      if (index !== -1) {
        this.alertas[index].animacao = '';
        this.cdr.markForCheck(); // Marca para verificação
      }
    }, 1500); // Duração da animação + pequeno buffer

    // Marca o componente para ser verificado pelo Angular
    this.cdr.markForCheck();

    // Opcional: Pulsar o gráfico real aqui
    // this.pulsarGraficoReal(novaOcorrencia.severity);
  }

  // Formata Ocorrencia -> Alerta (Reutilizável)
  formatarAlerta(
    oc: Ocorrencia,
    comAnimacao: boolean = false
  ): AlertaOcorrencia {
    return {
      hora: this.datePipe.transform(oc.start_ts, 'HH:mm:ss') || 'N/A',
      mensagem: oc.type || 'Ocorrência',
      tipo: oc.severity || 'Info (S)',
      animacao: comAnimacao ? 'aparecer' : '',
      origem: oc.category || 'API Detect',
    };
  }

  // === Funções Helper (Mantidas) ===
  isGrave(oc: Ocorrencia): boolean {
    // Considera graves as ocorrências com severidade A ou X
    const sev = (oc.severity || '').toString();
    return /\((A|X)\)/.test(sev);
  }
  getBadgeClass(tipo: string): string {
    switch (tipo) {
      case 'A':
        return 'badge-danger';
      case 'B':
        return 'badge-warning';
      case 'C':
        return 'badge-info';
      case 'X':
        return 'badge-dark';
      case 'P':
        return 'badge-secondary';
      default:
        return 'badge-secondary';
    }
  }
  getBadgeLetra(tipo: string): string {
    return tipo || '?';
  }
  getBadgeTexto(tipo: string): string {
    switch (tipo) {
      case 'A':
        return 'Grave';
      case 'B':
        return 'Médio';
      case 'C':
        return 'Leve';
      case 'X':
        return 'Gravíssimo';
      case 'P':
        return 'Placa';
      default:
        return 'Info';
    }
  }

  // === Lógica de Simulação (Manter Apenas a Parte Visual se Desejado) ===
  startLoop() {
    // Loop simples que empurra valores aleatórios para o gráfico para manter a UI "viva"
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
    this.loopInterval = setInterval(() => {
      try {
        const currentData = this.chartOptions.series[0].data as number[];
        const nextValue = Math.random() * 0.8; // valor baixo de tráfego
        const newData = [...currentData, Number(nextValue.toFixed(2))];
        if (newData.length > 50) newData.shift();
        this.chartOptions.series = [
          { ...this.chartOptions.series[0], data: newData },
        ];
        (this.chartOptions.xaxis as any).categories = newData.map((_, i) =>
          i.toString()
        );
        this.cdr.markForCheck();
      } catch (err) {
        console.error('Erro no loop do gráfico:', err);
      }
    }, 1500);
  }
  onPlayVideo() {
    // Remove simulação para transmissões ao vivo (tipo 'L') — evita mensagens "SIMULAÇÃO: undefined"
    console.warn('onPlayVideo: verificar se é simulação');
    const video = this.videos[this.videoIndex];
    if (!video) return;
    // Se for transmissão ao vivo, não mostramos alertas de simulação nem pulamos o gráfico
    if (video.tipo === 'L') {
      // marca como live se necessário
      this.isLive = true;
      this.cdr.markForCheck();
      return;
    }

    // Para demais tipos (vídeos de simulação), mostramos um alerta amigável com texto padrão
    let severidadeSimulada = 'Info (S)';
    if (video.tipo === 'A') severidadeSimulada = 'Grave (A)';

    const mensagens: Record<string, string> = {
      A: 'Repórter Parado',
      B: 'Evento Médio',
      C: 'Evento Leve',
    };
    const textoMensagem = mensagens[video.tipo] || 'Simulação de reprodução';

    const alertaSimulado = {
      hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
      mensagem: `SIMULAÇÃO: ${textoMensagem}`,
      tipo: severidadeSimulada,
      animacao: 'aparecer',
      origem: 'Player Local',
    };
    this.alertas.unshift(alertaSimulado);
    if (this.alertas.length > 10) this.alertas.pop();
    setTimeout(() => {
      const index = this.alertas.findIndex((a) => a === alertaSimulado);
      if (index !== -1) {
        this.alertas[index].animacao = '';
        this.cdr.markForCheck();
      }
    }, 1500);

    this.cdr.markForCheck(); // Atualiza UI para mostrar alerta simulado
    this.pulsarGraficoSimulado(video.tipo); // Chama pulso simulado
  }

  // Função separada para pulsar gráfico da simulação
  pulsarGraficoSimulado(tipoVideo: string): void {
    const valorFinal =
      tipoVideo === 'A' || tipoVideo === 'X' ? 3 : tipoVideo === 'B' ? 2 : 1;
    // ... (resto da lógica de animação do gráfico idêntica à anterior) ...
    let valorAtual = 0;
    const incremento = valorFinal / 20; // passosPulso
    if (this.loopInterval) clearInterval(this.loopInterval);

    const animInterval = setInterval(() => {
      valorAtual += incremento;
      if (valorAtual > valorFinal) valorAtual = valorFinal;
      const currentData = this.chartOptions.series[0].data as number[];
      const newData = [...currentData, Number(valorAtual.toFixed(2))];
      if (newData.length > 50) newData.shift(); // Limita tamanho do histórico no gráfico
      this.chartOptions.series = [
        { ...this.chartOptions.series[0], data: newData },
      ];
      (this.chartOptions.xaxis as any).categories = newData.map((_, i) =>
        i.toString()
      );
      this.cdr.markForCheck(); // Marca para atualizar o gráfico

      if (valorAtual >= valorFinal) {
        clearInterval(animInterval);
        setTimeout(() => {
          const currentDataReset = this.chartOptions.series[0].data as number[];
          const resetData = [...currentDataReset, 0];
          if (resetData.length > 50) resetData.shift();
          this.chartOptions.series = [
            { ...this.chartOptions.series[0], data: resetData },
          ];
          (this.chartOptions.xaxis as any).categories = resetData.map((_, i) =>
            i.toString()
          );
          this.cdr.markForCheck(); // Marca para atualizar
          this.startLoop(); // Reinicia loop normal
        }, 1500); // duracaoPulso
      }
    }, 50); // intervaloPulso
  }

  playNextVideo() {
    this.videoIndex = (this.videoIndex + 1) % this.videos.length;
    this.cdr.markForCheck();
  }
  trocarTransmissao(index: number) {
    if (index >= 0 && index < this.videos.length) {
      this.videoIndex = index;
      this.cdr.markForCheck();
    }
  }
}
