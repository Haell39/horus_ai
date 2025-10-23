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
import { CommonModule } from '@angular/common';

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
  templateUrl: './monitoramento.component.html',
  styleUrls: ['./monitoramento.component.css'],
})
export class MonitoramentoComponent implements OnInit, OnDestroy {
  @ViewChild('chart') chart!: ChartComponent;
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;

  // Video entries: optional `label` can override the default "Transmissão N" button text
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

  totalOcorrencias = 0;
  falhasGraves = 0;
  ultimaFalha: string | null = null;

  alertas: {
    hora: string;
    mensagem: string;
    tipo: string;
    animacao: string;
    origem: string;
  }[] = [];

  tempoContabilizacao = 3000;
  duracaoPulso = 1500;
  passosPulso = 20;
  intervaloPulso = 50;

  private loopInterval: any;

  public chartOptions: ChartOptions;

  constructor() {
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

  ngOnInit() {
    this.alertas = [
      {
        hora: new Date().toLocaleTimeString(),
        mensagem: 'Sistema iniciado com sucesso!',
        tipo: 'info',
        animacao: 'aparecer',
        origem: 'Sistema',
      },
    ];
    this.startLoop();
  }

  ngOnDestroy() {
    clearInterval(this.loopInterval);
  }

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
    this.totalOcorrencias++;

    // Atualiza última falha e contador de erros graves (A e X)
    if (video.tipo === 'A' || video.tipo === 'X') {
      this.ultimaFalha = video.tipo === 'A' ? 'Repórter Parado' : 'Freeze';
      this.falhasGraves++;
    } else if (video.tipo === 'B') {
      this.ultimaFalha = 'Áudio';
    } else if (video.tipo === 'C') {
      this.ultimaFalha = 'Fade';
    } else if (video.tipo === 'P') {
      // Placa de captura - treated as informational for now
      this.ultimaFalha = 'Placa de captura (info)';
    }

    // Adiciona alerta
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

    // Define intensidade do gráfico
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
        // Mantém o pulso curto antes de voltar a 0
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

          // Reinicia loop
          this.startLoop();

          // Após 30 segundos, "apaga" a linha do erro
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
    player.src = this.videos[this.videoIndex].src;
    player.pause();
  }
}
