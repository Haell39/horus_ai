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

import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';
import { WebsocketService } from '../../services/websocket.service'; // <<< IMPORT WebSocketService
import { Subscription } from 'rxjs'; // <<< IMPORT Subscription

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
  imports: [SidebarComponent, NgApexchartsModule, CommonModule],
  providers: [DatePipe],
  templateUrl: './monitoramento.component.html',
  styleUrls: ['./monitoramento.component.css'],
  // Usa OnPush para melhor performance com atualizações via WebSocket
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MonitoramentoComponent implements OnInit, OnDestroy {
  @ViewChild('chart') chart!: ChartComponent;
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;

  // Lista de vídeos de simulação (mantida)
  videos: Array<{
    src: string;
    tipo: string;
    descricao?: string;
    label?: string;
  }> = [
    {
      src: 'Ocorrência BDPE_Repórter Parado_27082025_08h14m03s.mp4',
      tipo: 'A',
      descricao: 'A - Grave',
    },
    { src: 'NE1_18_09_CORTE.mp4', tipo: 'B', descricao: 'B - Médio' },
    {
      src: 'Ocorrência BDPE_Corte de Sinal_22082025_08h29m02s.mp4',
      tipo: 'C',
      descricao: 'C - Leve',
    },
    {
      src: 'Ocorrência BDPE_Variação_20082025_06h32m59s.mp4',
      tipo: 'X',
      descricao: 'X - Gravíssimo',
    },
    {
      src: 'placa-captura-placeholder.mp4',
      tipo: 'P',
      descricao: 'Placa de captura',
      label: 'Placa de captura',
    },
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
    private cdr: ChangeDetectorRef // <<< INJETADO (Change Detector)
  ) {}

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
    console.warn(
      'Simulação onPlayVideo executada. Considerar remover lógica de atualização de dados daqui.'
    );
    const video = this.videos[this.videoIndex];
    // SIMULAÇÃO VISUAL: Adiciona alerta e pulsa gráfico, mas NÃO atualiza mais os contadores reais
    let severidadeSimulada = 'Info (S)';
    if (video.tipo === 'A') severidadeSimulada = 'Grave (A)';
    // ... (resto da lógica para severidadeSimulada) ...
    const mensagens: Record<string, string> = { A: 'Repórter Parado' /*...*/ };

    const alertaSimulado = {
      hora: this.datePipe.transform(new Date(), 'HH:mm:ss') || '',
      mensagem: `SIMULAÇÃO: ${mensagens[video.tipo]}`,
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
