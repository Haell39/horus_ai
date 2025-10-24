// frontend/srcÃ\app\pages\monitoramento\monitoramento.component.ts

import {
  Component,
  ViewChild,
  ElementRef,
  OnInit,
  OnDestroy,
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
} from 'ng-apexcharts';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { CommonModule, DatePipe } from '@angular/common'; // Importar DatePipe

// === NOVOS IMPORTS PARA API ===
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';

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

@Component({
  selector: 'app-monitoramento',
  standalone: true,
  imports: [SidebarComponent, NgApexchartsModule, CommonModule],
  providers: [DatePipe], // === ADICIONADO DatePipe como provider ===
  templateUrl: './monitoramento.component.html',
  styleUrls: ['./monitoramento.component.css'],
})
export class MonitoramentoComponent implements OnInit, OnDestroy {
  @ViewChild('chart') chart!: ChartComponent;
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;

  // A simulação de vídeo ainda está aqui (será removida na Fase 4)
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
    // ... (resto da sua lista de vídeos)
    {
      src: 'placa-captura-placeholder.mp4',
      tipo: 'P',
      descricao: 'Placa de captura',
      label: 'Placa de captura',
    },
  ];
  videoIndex = 0;

  // === Variáveis que serão preenchidas pela API ===
  totalOcorrencias = 0;
  falhasGraves = 0;
  ultimaFalha: string | null = null;
  alertas: {
    hora: string;
    mensagem: string;
    tipo: string; // Usaremos o 'severity' ou 'type' da API aqui
    animacao: string;
    origem: string;
  }[] = [];
  // ===============================================

  tempoContabilizacao = 3000;
  duracaoPulso = 1500;
  passosPulso = 20;
  intervaloPulso = 50;

  private loopInterval: any;
  public chartOptions: ChartOptions;

  // === CONSTRUTOR ATUALIZADO ===
  constructor(
    private ocorrenciaService: OcorrenciaService, // Injeta o serviço da API
    private datePipe: DatePipe // Injeta o DatePipe para formatar datas
  ) {
    this.chartOptions = {
      // ... (configuração do gráfico mantida igual) ...
    };
  }

  // === ngOnInit ATUALIZADO ===
  ngOnInit() {
    // 1. Adiciona o alerta inicial de "Sistema iniciado"
    this.alertas = [
      {
        hora: new Date().toLocaleTimeString(),
        mensagem: 'Sistema iniciado com sucesso!',
        tipo: 'info',
        animacao: 'aparecer',
        origem: 'Sistema',
      },
    ];

    // 2. Chama a nova função para carregar dados da API
    this.carregarDadosIniciais();

    // 3. Inicia o loop do gráfico (lógica original)
    this.startLoop();
  }

  // === NOVA FUNÇÃO PARA CARREGAR DADOS DA API ===
  carregarDadosIniciais() {
    console.log('MonitoramentoComponent: Buscando dados iniciais da API...');
    this.ocorrenciaService.getOcorrencias().subscribe(
      (data: Ocorrencia[]) => {
        console.log(`Dados recebidos: ${data.length} ocorrências.`);

        if (data.length > 0) {
          // Atualiza os cards
          this.totalOcorrencias = data.length;

          // Define "falha grave" (Ex: severity 'Alta' ou 'Média'. Ajuste conforme sua regra)
          this.falhasGraves = data.filter(
            (oc) => oc.severity === 'Alta' || oc.severity === 'Média'
          ).length;

          // Pega a ocorrência MAIS RECENTE (primeira da lista, pois ordenamos por ID desc)
          const maisRecente = data[0];
          this.ultimaFalha = maisRecente.type || 'N/A'; // Ex: "Ruído / chiado"

          // Mapeia os dados da API para o formato dos Alertas
          const alertasDaApi = data.map((oc: Ocorrencia) => {
            return {
              hora: this.datePipe.transform(oc.start_ts, 'HH:mm:ss') || 'N/A',
              mensagem: oc.type || 'Ocorrência', // Ex: "Ruído / chiado"
              tipo: oc.severity || 'info', // Ex: "Média"
              animacao: 'aparecer', // Animação de entrada
              origem: oc.category || 'API', // Ex: "Áudio Técnico"
            };
          });

          // Adiciona os alertas da API na lista (mantendo o "Sistema iniciado")
          // Usamos slice(0, 4) para pegar apenas os 4 mais recentes da API
          this.alertas = [...this.alertas, ...alertasDaApi.slice(0, 4)];
        }
      },
      (error) => {
        console.error('Erro ao buscar dados para o monitoramento:', error);
        // Adiciona um alerta de erro
        this.alertas.unshift({
          hora: new Date().toLocaleTimeString(),
          mensagem: 'Falha ao conectar com a API',
          tipo: 'erro', // (você pode estilizar a classe 'erro')
          animacao: 'aparecer',
          origem: 'Sistema',
        });
      }
    );
  }

  ngOnDestroy() {
    clearInterval(this.loopInterval);
  }

  // === LÓGICA DE SIMULAÇÃO (Mantida por enquanto) ===
  // ... (startLoop, onPlayVideo, playNextVideo, trocarTransmissao) ...
  // ... (Todo o resto do seu arquivo .ts permanece igual) ...
  startLoop() {
    this.loopInterval = setInterval(() => {
      const newData = [
        ...this.chartOptions.series[0].data.map((v) => Number(v)),
        0,
      ];
      this.chartOptions.series = [
        { ...this.chartOptions.series[0], data: newData },
      ];
      this.chartOptions.xaxis.categories = newData.map((_, i) => i.toString());
    }, 1000);
  }

  onPlayVideo() {
    const video = this.videos[this.videoIndex];
    // NOTA: Esta linha agora vai ADICIONAR ao total que veio da API.
    this.totalOcorrencias++;

    if (video.tipo === 'A' || video.tipo === 'X') {
      this.ultimaFalha = video.tipo === 'A' ? 'Repórter Parado' : 'Freeze';
      // Esta linha também vai ADICIONAR ao total que veio da API.
      this.falhasGraves++;
    } else if (video.tipo === 'B') {
      this.ultimaFalha = 'Áudio';
    } else if (video.tipo === 'C') {
      this.ultimaFalha = 'Fade';
    } else if (video.tipo === 'P') {
      this.ultimaFalha = 'Placa de captura (info)';
    }

    const mensagens: Record<string, string> = {
      A: 'Repórter Parado',
      B: 'Problema de Áudio',
      C: 'Fade detectado',
      X: 'Freeze detectado',
      P: 'Placa de captura selecionada',
    };
    this.alertas.unshift({
      hora: new Date().toLocaleTimeString(),
      mensagem: mensagens[video.tipo],
      tipo: video.tipo,
      animacao: 'aparecer',
      origem: 'Jornal Nacional',
    });
    this.alertas = this.alertas.slice(0, 4);
    setTimeout(() => this.alertas.forEach((a) => (a.animacao = '')), 1000);
    if (this.alertas.length > 6) this.alertas.pop();

    const valorFinal =
      video.tipo === 'A' || video.tipo === 'X' ? 3 : video.tipo === 'B' ? 2 : 1;

    let valorAtual = 0;
    const incremento = valorFinal / this.passosPulso;
    clearInterval(this.loopInterval);

    const animInterval = setInterval(() => {
      valorAtual += incremento;
      if (valorAtual > valorFinal) valorAtual = valorFinal;

      const newData = [
        ...this.chartOptions.series[0].data.map((v) => Number(v)),
        Number(valorAtual.toFixed(2)),
      ];
      this.chartOptions.series = [
        { ...this.chartOptions.series[0], data: newData },
      ];
      this.chartOptions.xaxis.categories = newData.map((_, i) => i.toString());

      if (valorAtual >= valorFinal) {
        clearInterval(animInterval);
        setTimeout(() => {
          const resetData = [
            ...this.chartOptions.series[0].data.map((v) => Number(v)),
            0,
          ];
          this.chartOptions.series = [
            { ...this.chartOptions.series[0], data: resetData },
          ];
          this.chartOptions.xaxis.categories = resetData.map((_, i) =>
            i.toString()
          );
          this.startLoop();
          setTimeout(() => {
            const clearedData = this.chartOptions.series[0].data.map(() => 0);
            this.chartOptions.series = [
              { ...this.chartOptions.series[0], data: clearedData },
            ];
            this.chartOptions.xaxis.categories = clearedData.map((_, i) =>
              i.toString()
            );
          }, 30000); // 30 segundos
        }, this.duracaoPulso);
      }
    }, this.intervaloPulso);
  }

  playNextVideo() {
    if (this.videoIndex < this.videos.length - 1) this.videoIndex++;
    else this.videoIndex = 0;

    const player = this.videoPlayer.nativeElement;
    player.src = this.videos[this.videoIndex].src;
    player.pause();
  }

  trocarTransmissao(index: number) {
    this.videoIndex = index;
    const player = this.videoPlayer.nativeElement;
    Mplayer.src = this.videos[this.videoIndex].src;
    player.pause();
  }
}
