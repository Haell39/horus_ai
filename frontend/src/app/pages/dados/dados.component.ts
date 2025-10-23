import { Component, ViewChild, AfterViewInit } from '@angular/core';
import { SidebarComponent } from '../../shared/sidebar/sidebar.component';
import { FormsModule } from '@angular/forms';
import {
  NgApexchartsModule,
  ChartComponent,
  ApexAxisChartSeries,
  ApexNonAxisChartSeries,
  ApexChart,
  ApexDataLabels,
  ApexPlotOptions,
  ApexYAxis,
  ApexLegend,
  ApexStroke,
  ApexXAxis,
  ApexFill,
  ApexTooltip,
  ApexResponsive
} from 'ng-apexcharts';

export type BarChartOptions = {
  series: ApexAxisChartSeries;
  chart: ApexChart;
  dataLabels: ApexDataLabels;
  plotOptions: ApexPlotOptions;
  yaxis: ApexYAxis;
  xaxis: ApexXAxis;
  fill: ApexFill;
  tooltip: ApexTooltip;
  stroke: ApexStroke;
  legend: ApexLegend;
};

export type DonutChartOptions = {
  series: ApexNonAxisChartSeries;
  chart: ApexChart;
  labels: any;
  responsive: ApexResponsive[];
  dataLabels?: ApexDataLabels;
  legend?: ApexLegend;
  colors?: string[];
};

export type HorizontalBarOptions = {
  series: ApexAxisChartSeries;
  chart: ApexChart;
  dataLabels: ApexDataLabels;
  plotOptions: ApexPlotOptions;
  xaxis: ApexXAxis;
  yaxis: ApexYAxis;
  fill?: ApexFill;
  legend?: ApexLegend;
};

@Component({
  selector: 'app-dados',
  standalone: true,
  imports: [SidebarComponent, NgApexchartsModule, FormsModule],
  templateUrl: './dados.component.html',
  styleUrls: ['./dados.component.css']
})
export class DadosComponent implements AfterViewInit {
  @ViewChild('barChart') barChart!: ChartComponent;
  @ViewChild('donutChart') donutChart!: ChartComponent;
  @ViewChild('horizontalChart') horizontalChart!: ChartComponent;

  public barChartOptions!: Partial<BarChartOptions>;
  public donutChartOptions!: Partial<DonutChartOptions>;
  public horizontalChartOptions!: Partial<HorizontalBarOptions>;

  // === Filtros ===
  periodoSelecionado = '7d';
  tipoErroSelecionado = 'todos';

  // 🎨 Paleta fixa
  private coresPadrao = ['#FF4B4B', '#4B79FF', '#FFD700', '#00FF00', '#FF00FF', 'gray', '#FFFFFF'];

  constructor() {
    this.carregarGraficos();
  }

  ngAfterViewInit() {
    // Atualiza cores quando o tema muda
    const observer = new MutationObserver(() => this.atualizarCoresTema());
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  }

  // ===========================
  // Detecta tema atual
  // ===========================
  private getTemaAtual(): 'light' | 'dark' {
    return document.body.classList.contains('light-theme') ? 'light' : 'dark';
  }

  private getCorTexto(): string {
    return this.getTemaAtual() === 'light' ? '#222222' : '#ffffff';
  }

  private getCoresDonut(): string[] {
    return this.getTemaAtual() === 'light'
      ? ['#FF4B4B', '#4B79FF', '#FFD700', '#00FF00', '#FF00FF', 'gray', '#333333']
      : this.coresPadrao;
  }

  // ===========================
  // Carrega gráficos
  // ===========================
  carregarGraficos() {
    const corTexto = this.getCorTexto();
    const coresDonut = this.getCoresDonut();

    // 🔵 Gráfico Vertical
    this.barChartOptions = {
      series: [
        { name: 'Erro Grave', data: [5, 8, 4, 7, 6, 9, 5, 6, 8] },
        { name: 'Total de Falhas', data: [12, 15, 10, 18, 14, 20, 16, 18, 19] }
      ],
      chart: { type: 'bar', height: 500, width: 830, animations: { enabled: false } },
      plotOptions: { bar: { horizontal: false, columnWidth: '55%', borderRadius: 2 } },
      dataLabels: { enabled: false },
      stroke: { show: true, width: 2, colors: ['transparent'] },
      xaxis: { categories: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'], labels: { style: { colors: Array(9).fill(corTexto), fontSize: '14px' } } },
      yaxis: { labels: { style: { colors: Array(5).fill(corTexto), fontSize: '14px' } } },
      fill: { opacity: 1, colors: ['#FF4B4B', '#4B79FF'] },
      tooltip: { y: { formatter: (val: number) => val + ' erros' } },
      legend: { position: 'bottom', labels: { colors: corTexto } }
    };

    // 🟣 Gráfico Donut
    this.donutChartOptions = {
      series: [44, 55, 13, 43, 22, 10, 9],
      chart: { type: 'donut', height: 350, width: 350, animations: { enabled: false } },
      labels: ['Fade', 'Freeze', 'LipSync', 'Imagem errada', 'Reporte parado', 'Áudio', 'Variação'],
      colors: coresDonut,
      dataLabels: { style: { colors: Array(7).fill(corTexto) } },
      legend: { labels: { colors: corTexto }, markers: { fillColors: coresDonut } },
      responsive: [{ breakpoint: 480, options: { chart: { width: 400 }, legend: { position: 'bottom' } } }]
    };

    // 🟢 Gráfico Horizontal
    this.horizontalChartOptions = {
      series: [{ name: 'Ocorrências', data: [10, 20, 30, 35, 40] }],
      chart: { type: 'bar', height: 200, width: 600, animations: { enabled: false } },
      plotOptions: { bar: { horizontal: true } },
      dataLabels: { enabled: false },
      xaxis: { categories: ['Fade','Freeze','LipSync','Imagem errada','Reporte parado'], labels: { style: { colors: Array(5).fill(corTexto), fontSize: '13px' } } },
      yaxis: { labels: { style: { colors: Array(5).fill(corTexto), fontSize: '13px' } } },
      fill: { opacity: 1, colors: ['#4B79FF'] },
      legend: { position: 'right', labels: { colors: corTexto }, markers: { width: 12, height: 12, fillColors: ['#4B79FF'] } }
    };
  }

  // ===========================
  // Atualiza cores quando o tema muda
  // ===========================
  atualizarCoresTema() {
    const corTexto = this.getCorTexto();
    const coresDonut = this.getCoresDonut();

    this.barChart.updateOptions({
      xaxis: { labels: { style: { colors: Array(9).fill(corTexto) } } },
      yaxis: { labels: { style: { colors: Array(5).fill(corTexto) } } },
      legend: { labels: { colors: corTexto } }
    });

    this.donutChart.updateOptions({
      colors: coresDonut,
      dataLabels: { style: { colors: Array(7).fill(corTexto) } },
      legend: { labels: { colors: corTexto }, markers: { fillColors: coresDonut } }
    });

    this.horizontalChart.updateOptions({
      xaxis: { labels: { style: { colors: Array(5).fill(corTexto) } } },
      yaxis: { labels: { style: { colors: Array(5).fill(corTexto) } } },
      legend: { labels: { colors: corTexto } }
    });
  }

  // ===========================
  // Aplica filtros
  // ===========================
  aplicarFiltros() {
    let donutData: number[] = [];
    let horizontalData: number[] = [];

    switch (this.periodoSelecionado) {
      case '1d':
        donutData = [10, 15, 5, 8, 4, 3, 2];
        horizontalData = [5, 10, 15, 18, 20];
        break;
      case '7d':
        donutData = [20, 25, 10, 15, 8, 5, 3];
        horizontalData = [10, 20, 30, 35, 40];
        break;
      case '30d':
        donutData = [40, 50, 20, 30, 25, 15, 10];
        horizontalData = [15, 25, 35, 40, 45];
        break;
      case '2m':
        donutData = [50, 60, 25, 35, 30, 20, 10];
        horizontalData = [18, 28, 45, 50, 55];
        break;
      default:
        donutData = [20, 25, 10, 15, 8, 5, 3];
        horizontalData = [10, 20, 30, 35, 40];
    }

    if (this.tipoErroSelecionado === 'grave') {
      donutData = [donutData[0], 0, 0, 0, 0, 0, 0];
    } else if (this.tipoErroSelecionado === 'simples') {
      donutData = [0, donutData[1], donutData[2], donutData[3], donutData[4], donutData[5], donutData[6]];
    }

    this.donutChart.updateOptions({ series: donutData, colors: this.getCoresDonut() });
    this.horizontalChart.updateSeries([{ name: 'Ocorrências', data: horizontalData }]);
  }
}
