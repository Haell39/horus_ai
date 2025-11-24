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
import { Subscription, Subject } from 'rxjs'; // <<< IMPORT Subscription, Subject
import { auditTime } from 'rxjs/operators';
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
  colors?: string[];
};

// Interface interna para Alertas (mantida)
interface AlertaOcorrencia {
  hora: string;
  mensagem: string;
  tipo: string;
  animacao: string;
  origem: string;
  ts?: string; // ISO timestamp para persistência/ordenacao
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
  // optional custom SRT URL
  public customSrtUrl: string = '';

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
  public debugAnalysis: boolean = false;
  public uploadingFile: boolean = false;
  // Simple toast notifications (ephemeral, non-blocking)
  public toasts: Array<{
    id: number;
    message: string;
    type?: 'info' | 'success' | 'error';
  }> = [];

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

  // --- Time-series buckets (Option A) ---
  private bucketSizeMs = 1000; // 1s buckets by default
  private bucketCount = 300; // last 5 minutes (300 * 1s)
  private buckets: Array<{ ts: number; counts: number[] }> = [];
  private incomingEvents$ = new Subject<Ocorrencia>();
  private incomingSubscription: Subscription | null = null;
  // Set to deduplicate occurrences (prevents double counting from upload + ws)
  private seenOccurrenceKeys = new Set<string>();

  // <<< VARIÁVEIS WebSocket >>>
  private wsSubscription: Subscription | null = null;
  public isWsConnected: boolean = false; // Status da conexão WS
  // Chave localStorage para alertas recentes
  private ALERTAS_STORAGE_KEY = 'horus_alertas_recentes_v1';
  // Quantos alertas mostrar no card (mais recente primeiro)
  private MAX_ALERTAS_NA_TELA = 7;

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
    // Carrega alertas persistidos (se houver) ou cria um alerta inicial
    this.alertas = this.loadAlertasFromStorage();
    if (!this.alertas || this.alertas.length === 0) {
      const inicial: AlertaOcorrencia = {
        hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
        mensagem: 'Sistema iniciado.',
        tipo: 'Info (S)',
        animacao: 'aparecer',
        origem: 'Sistema',
        ts: new Date().toISOString(),
      };
      this.addAlerta(inicial);
    }

    // initialize live buffer and bucketed chart BEFORE backfill
    this.initBuckets(Date.now());
    this.carregarDadosIniciais(); // Carga HTTP
    // subscribe to buffered incoming events (batch updates)
    this.incomingSubscription = this.incomingEvents$
      .pipe(auditTime(500))
      .subscribe(() => {
        this.processBufferedEvents();
        this.updateChartFromBuckets();
        this.cdr.markForCheck();
      });
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
    this.ocorrenciaService
      .uploadAnalysis(this.selectedFile, fps, this.debugAnalysis)
      .subscribe({
        next: (res: any) => {
          console.log('Upload analysis response:', res);
          if (res && res.diagnostic) {
            console.log('Diagnostic data (per-frame top-k):', res.diagnostic);
            this.showToast(
              'Diagnóstico gerado — ver console do navegador para detalhes.',
              'info',
              6000
            );
          }
          // If backend returned a created occurrence object (sync), process it like a new occurrence
          if (res && res.id) {
            this.processarNovaOcorrencia(res as any);
          } else if (res && res.status === 'ok') {
            // Notificação imediata (não persistir nos "Alertas Recentes" para casos normais)
            this.showToast(
              res.message || 'Arquivo analisado — sem falhas detectadas.',
              'success'
            );
          } else if (res && res.status === 'queued') {
            // Inform user that processing is queued (ephemeral notification only)
            this.showToast(
              res.message || 'Arquivo enviado. Processamento em segundo plano.',
              'info'
            );
          }
          this.selectedFile = null;
          this.uploadingFile = false;
          this.debugAnalysis = false;
          this.cdr.markForCheck();
        },
        error: (err) => {
          console.error('Erro ao enviar arquivo para análise:', err);
          this.addAlerta({
            hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
            mensagem: 'Falha ao enviar arquivo para análise',
            tipo: 'Erro (E)',
            animacao: 'aparecer',
            origem: 'Análise de Arquivo',
            ts: new Date().toISOString(),
          });
          this.uploadingFile = false;
          this.cdr.markForCheck();
        },
      });
  }

  // Toast helpers
  showToast(
    message: string,
    type: 'info' | 'success' | 'error' = 'info',
    timeout = 3500
  ) {
    try {
      const id = Date.now() + Math.floor(Math.random() * 1000);
      this.toasts.push({ id, message, type });
      this.cdr.markForCheck();
      // play notification sound according to configuration
      try {
        this.playSoundForToast(type);
      } catch (e) {
        // noop
      }

      setTimeout(() => {
        this.toasts = this.toasts.filter((t) => t.id !== id);
        this.cdr.markForCheck();
      }, timeout);
    } catch (e) {}
  }

  // Plays a sound based on saved configuration and toast type
  private playSoundForToast(type: 'info' | 'success' | 'error') {
    try {
      const raw = localStorage.getItem('configuracoes');
      const cfg = raw ? JSON.parse(raw) : null;
      const soundPref = cfg?.somAlerta || 'beep';
      const map: Record<string, string> = {
        beep: 'Alerta-Curto.mp3',
        digital: 'Alerta-Digital.mp3',
        alerta: 'Alerta-Sutil.mp3',
      };
      // For errors we prefer the 'digital' alert to be more noticeable
      let file = map[soundPref] || map['beep'];
      let volume = 0.8;
      if (type === 'error') {
        file = map['digital'] || file;
        volume = 1.0;
      }
      const audio = new Audio(file);
      audio.volume = volume;
      audio.play().catch(() => {});
    } catch (e) {
      // ignore play errors
    }
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
    if (this.incomingSubscription) {
      this.incomingSubscription.unsubscribe();
      this.incomingSubscription = null;
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
      // if user provided a custom URL, use it; otherwise fall back to streamId/env
      if (this.customSrtUrl && this.customSrtUrl.trim().length > 0) {
        payload.url = this.customSrtUrl.trim();
      } else {
        payload.streamId = environment.streamId;
      }
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
      // Start disabled; we enable it when we have series data.
      tooltip: {
        enabled: false,
        theme: 'dark',
        style: { fontSize: '12px', fontFamily: 'Segoe UI' },
      },
      // Series colors (top-level Apex option) - match severity ordering: X, A, B, C
      colors: ['#e53935', '#FFC107', '#1E88E5', '#43A047'],
    };
  }

  // Carrega dados históricos via HTTP
  carregarDadosIniciais() {
    console.log('MonitoramentoComponent: Buscando dados iniciais HTTP...');
    // Primeiro: solicita ao backend o total real de ocorrências (para o card)
    this.ocorrenciaService.getTotalOcorrencias().subscribe({
      next: (res: any) => {
        try {
          this.totalOcorrencias = Number(res?.count || 0);
        } catch (e) {
          console.warn('Erro ao ler total de ocorrências:', e);
        }
        this.cdr.markForCheck();
      },
      error: (err) => {
        console.warn('Falha ao obter total de ocorrências:', err);
      },
    });

    // Pega apenas os últimos 50 (ou menos) para a carga inicial
    this.ocorrenciaService.getOcorrencias().subscribe({
      // Usando objeto Observer
      next: (data: Ocorrencia[]) => {
        console.log(`HTTP: Recebidas ${data.length} ocorrências.`);
        if (data.length > 0) {
          // Não sobrescrever o total obtido via /ocorrencias/count; usar apenas como fallback
          if (!this.totalOcorrencias || this.totalOcorrencias === 0) {
            this.totalOcorrencias = data.length; // Pode precisar buscar o total real se API paginar
          }
          this.falhasGraves = data.filter((oc) => this.isGrave(oc)).length; // Calcula sobre todos os dados recebidos
          this.ultimaFalha = data[0].type || 'N/A'; // Assume que API retorna mais recente primeiro

          const MAX_ALERTAS_INICIAIS = 5; // Mostra os 5 mais recentes
          const alertasDaApi = data
            .slice(0, MAX_ALERTAS_INICIAIS)
            .map((oc) => this.formatarAlerta(oc)) // Formata sem animação
            .reverse(); // Inverte para mostrar mais antigo primeiro na carga inicial

          // Adiciona os históricos APÓS o alerta de "Sistema iniciado"
          // Garantir que a lista final respeite o limite e a ordenação por ts (mais recente primeiro)
          this.alertas = [...this.alertas, ...alertasDaApi];
          // Ordena por ts desc e limita para MAX_ALERTAS_NA_TELA
          this.alertas.sort((a, b) => {
            const ta = a.ts ? Date.parse(a.ts) : 0;
            const tb = b.ts ? Date.parse(b.ts) : 0;
            return tb - ta;
          });
          if (this.alertas.length > this.MAX_ALERTAS_NA_TELA) {
            this.alertas = this.alertas.slice(0, this.MAX_ALERTAS_NA_TELA);
          }
          // Persiste a versão truncada
          this.saveAlertasToStorage();

          this.cdr.markForCheck(); // Marca para verificação com OnPush

          // --- Backfill into buckets for Option A charting ---
          try {
            // API may return newest-first; apply oldest -> newest so counts accumulate
            for (let i = data.length - 1; i >= 0; i--) {
              const oc = data[i];
              const key = this.getOccurrenceKey(oc);
              if (!this.seenOccurrenceKeys.has(key)) {
                this.applyOccurrenceToBuckets(oc);
                this.seenOccurrenceKeys.add(key);
              }
            }
            this.updateChartFromBuckets();
            this.cdr.markForCheck();
          } catch (e) {
            console.warn('Erro ao popular histórico no gráfico:', e);
          }
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
    try {
      console.log(
        'Processando nova ocorrência WS:',
        novaOcorrencia?.id,
        novaOcorrencia?.type
      );

      // Deduplicação local: se já vimos essa ocorrência (backfill ou upload),
      // evitamos re-contar/duplicar no gráfico e nos totais.
      const ocKey = this.getOccurrenceKey(novaOcorrencia);
      const alreadySeen = this.seenOccurrenceKeys.has(ocKey);
      if (!alreadySeen) {
        this.seenOccurrenceKeys.add(ocKey);
        // Atualiza contadores simples
        this.totalOcorrencias = (this.totalOcorrencias || 0) + 1;
        if (this.isGrave(novaOcorrencia))
          this.falhasGraves = (this.falhasGraves || 0) + 1;
      }
      this.ultimaFalha = novaOcorrencia.type || 'N/A';

      // Formata e adiciona alerta com animação
      const novoAlerta = this.formatarAlerta(novaOcorrencia, true);
      this.addAlerta(novoAlerta);
      setTimeout(() => {
        const index = this.alertas.findIndex(
          (a) => a === novoAlerta || a.ts === novoAlerta.ts
        );
        if (index !== -1) {
          this.alertas[index].animacao = '';
          this.cdr.markForCheck();
        }
      }, 1500);

      // Atualiza buckets e notifica o buffer para re-render do gráfico em lote
      if (!alreadySeen) {
        this.applyOccurrenceToBuckets(novaOcorrencia);
        this.incomingEvents$.next(novaOcorrencia);
      } else {
        console.log(
          'Ocorrência já vista localmente, ignorando incremento de buckets:',
          ocKey
        );
      }

      // Opção: tocar som para severidades altas (placeholder)
      const sev = (novaOcorrencia.severity || '').toString();
      if (
        sev.match(/X|Gravíssim|Gravíssima|\(X\)/i) ||
        sev.match(/\(A\)|\bA\b|Grave/i)
      ) {
        // Som simples: usar API de áudio se houver arquivo, aqui apenas log
        console.log('Alerta severo recebido, severidade:', sev);
      }

      this.cdr.markForCheck();
    } catch (e) {
      console.warn('Erro ao processar nova ocorrência:', e);
    }
  }

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

  // ---- Bucket helpers for Option A (time-series counts) ----
  private initBuckets(nowMs: number) {
    this.buckets = [];
    const start = nowMs - this.bucketSizeMs * (this.bucketCount - 1);
    for (let i = 0; i < this.bucketCount; i++) {
      this.buckets.push({
        ts: start + i * this.bucketSizeMs,
        counts: [0, 0, 0, 0],
      });
    }
  }

  private processBufferedEvents() {
    // Placeholder: events are applied when they arrive via incomingEvents$.next
    // We use auditTime to batch UI updates and call updateChartFromBuckets()
  }

  private applyOccurrenceToBuckets(oc: Ocorrencia) {
    try {
      const t = Date.parse(oc.start_ts as any);
      if (isNaN(t)) return;
      const now = Date.now();
      // If occurrence is older than window start, ignore
      const windowStart = now - this.bucketSizeMs * (this.bucketCount - 1);
      if (t < windowStart) {
        return;
      }
      // ensure bucket range
      this.ensureBucketsCover(now);
      const idx = Math.floor((t - this.buckets[0].ts) / this.bucketSizeMs);
      if (idx < 0 || idx >= this.buckets.length) return;
      const sevIdx = this.mapSeverityToIndex(oc.severity || '');
      this.buckets[idx].counts[sevIdx] += 1;
    } catch (e) {
      // ignore
    }
  }

  private ensureBucketsCover(nowMs: number) {
    const expectedStart = nowMs - this.bucketSizeMs * (this.bucketCount - 1);
    const delta = expectedStart - this.buckets[0].ts;
    if (delta <= 0) return;
    const shiftBuckets = Math.floor(delta / this.bucketSizeMs);
    if (shiftBuckets >= this.bucketCount) {
      this.initBuckets(nowMs);
      return;
    }
    for (let s = 0; s < shiftBuckets; s++) {
      this.buckets.shift();
      const lastTs = this.buckets[this.buckets.length - 1].ts;
      this.buckets.push({
        ts: lastTs + this.bucketSizeMs,
        counts: [0, 0, 0, 0],
      });
    }
  }

  private mapSeverityToIndex(sev: string): number {
    const s = (sev || '').toString();
    if (/X|Gravíssim|Gravíssima|Gravissimo/i.test(s)) return 0;
    if (/\(A\)|\bA\b|Grave/i.test(s)) return 1;
    if (/\(B\)|\bB\b|Médio|Medio/i.test(s)) return 2;
    if (/\(C\)|\bC\b|Leve/i.test(s)) return 3;
    return 3;
  }

  private updateChartFromBuckets() {
    try {
      const labels = this.buckets.map((b) =>
        new Date(b.ts).toLocaleTimeString()
      );
      const series = [
        { name: 'Gravíssimo (X)', data: this.buckets.map((b) => b.counts[0]) },
        { name: 'Grave (A)', data: this.buckets.map((b) => b.counts[1]) },
        { name: 'Médio (B)', data: this.buckets.map((b) => b.counts[2]) },
        { name: 'Leve (C)', data: this.buckets.map((b) => b.counts[3]) },
      ];
      this.chartOptions = {
        ...this.chartOptions,
        series: series as any,
        labels: labels,
        xaxis: { ...(this.chartOptions.xaxis as any), categories: labels },
        tooltip: {
          ...((this.chartOptions && this.chartOptions.tooltip) as any),
          enabled: true,
        },
        legend: { show: true },
      };
    } catch (e) {
      console.warn('Erro ao atualizar gráfico a partir dos buckets:', e);
    }
  }

  // Gera uma chave estável para uma ocorrência para deduplicação local
  private getOccurrenceKey(oc: Ocorrencia): string {
    try {
      if (!oc) return 'unknown';
      if ((oc as any).id) return `id:${(oc as any).id}`;
      const ts = oc.start_ts || oc.created_at || '';
      const type = oc.type || '';
      const sev = oc.severity || '';
      const clip = (oc.evidence && (oc.evidence as any).clip_path) || '';
      return `${ts}|${type}|${sev}|${clip}`;
    } catch (e) {
      return 'unknown';
    }
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

  // --- Alert persistence helpers ---
  private loadAlertasFromStorage(): AlertaOcorrencia[] {
    try {
      const raw = localStorage.getItem(this.ALERTAS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as AlertaOcorrencia[];
      // Ordena por timestamp (desc)
      parsed.sort((a, b) => {
        const ta = a.ts ? Date.parse(a.ts) : 0;
        const tb = b.ts ? Date.parse(b.ts) : 0;
        return tb - ta;
      });
      // Limitamos para evitar crescimento indefinido ao carregar
      return parsed.slice(0, this.MAX_ALERTAS_NA_TELA);
    } catch (e) {
      console.warn('Falha ao carregar alertas do storage:', e);
      return [];
    }
  }

  private saveAlertasToStorage(): void {
    try {
      const MAX = 50; // limite de armazenamento local
      const toSave = this.alertas.slice(0, MAX);
      localStorage.setItem(this.ALERTAS_STORAGE_KEY, JSON.stringify(toSave));
    } catch (e) {
      console.warn('Falha ao salvar alertas no storage:', e);
    }
  }

  private addAlerta(a: AlertaOcorrencia): void {
    try {
      if (!a.ts) a.ts = new Date().toISOString();
      if (!a.hora)
        a.hora = this.datePipe.transform(new Date(a.ts), 'HH:mm:ss') || '';
      // adiciona no início
      this.alertas.unshift(a);
      // ordena por ts desc
      this.alertas.sort((x, y) => {
        const tx = x.ts ? Date.parse(x.ts) : 0;
        const ty = y.ts ? Date.parse(y.ts) : 0;
        return ty - tx;
      });
      // limita a quantidade exibida em tela
      if (this.alertas.length > this.MAX_ALERTAS_NA_TELA)
        this.alertas = this.alertas.slice(0, this.MAX_ALERTAS_NA_TELA);
      // persiste
      this.saveAlertasToStorage();
      // animação temporária
      if (a.animacao) {
        setTimeout(() => {
          const idx = this.alertas.findIndex(
            (it) => it === a || it.ts === a.ts
          );
          if (idx !== -1) {
            this.alertas[idx].animacao = '';
            this.cdr.markForCheck();
          }
        }, 1500);
      }
      this.cdr.markForCheck();
    } catch (e) {
      console.warn('addAlerta erro:', e);
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
    this.addAlerta({ ...alertaSimulado, ts: new Date().toISOString() });
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
