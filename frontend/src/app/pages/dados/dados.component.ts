// frontend/src/app/pages/dados/dados.component.ts
// (Versão 5 - Correção dos erros de Tipo)

import { Component, ViewChild, AfterViewInit, OnInit } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
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
  ApexResponsive,
} from 'ng-apexcharts';

import { Observable, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia';

// (Definição dos Tipos de Gráfico mantida)
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
  imports: [SidebarComponent, NgApexchartsModule, FormsModule, CommonModule],
  providers: [DatePipe],
  templateUrl: './dados.component.html',
  styleUrls: ['./dados.component.css'],
})
export class DadosComponent implements AfterViewInit, OnInit {
  @ViewChild('barChart') barChart!: ChartComponent;
  @ViewChild('donutChart') donutChart!: ChartComponent;
  @ViewChild('horizontalChart') horizontalChart!: ChartComponent;

  // === CORREÇÃO DE TIPO (1/3) ===
  // Removemos o 'Partial' e usamos '!' para dizer ao TS que vamos inicializar
  public barChartOptions!: BarChartOptions;
  public donutChartOptions!: DonutChartOptions;
  public horizontalChartOptions!: HorizontalBarOptions;

  periodoSelecionado = '7d';
  tipoErroSelecionado = 'todos';

  private coresPadrao = [
    '#FF4B4B',
    '#4B79FF',
    '#FFD700',
    '#00FF00',
    '#FF00FF',
    'gray',
    '#FFFFFF',
  ];

  public ocorrencias$: Observable<Ocorrencia[]> = of([]);
  public errorMsg: string | null = null;
  public dadosCarregados: Ocorrencia[] = [];

  constructor(
    private ocorrenciaService: OcorrenciaService,
    private datePipe: DatePipe
  ) {}

  ngOnInit(): void {
    console.log('DadosComponent: ngOnInit - Carregando...');
    this.carregarEstruturaGraficos(); // Primeiro inicializa a estrutura
    this.carregarDadosApi(); // Depois busca os dados
  }

  ngAfterViewInit() {
    const observer = new MutationObserver(() => this.atualizarCoresTema());
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['class'],
    });
  }

  carregarDadosApi(): void {
    this.errorMsg = null;
    this.ocorrencias$ = this.ocorrenciaService.getOcorrencias().pipe(
      tap((data) => {
        console.log(`Dados recebidos da API: ${data.length} ocorrências.`);
        this.dadosCarregados = data;
        this.aplicarFiltros(); // Chama a função de filtro (que atualiza os gráficos)
      }),
      catchError((err) => {
        console.error('Erro ao buscar ocorrências:', err);
        this.errorMsg =
          'Falha ao carregar dados da API. O backend está rodando?';
        this.dadosCarregados = [];
        return of([]);
      })
    );
  }

  // ===================================
  // === ATUALIZAÇÃO DOS GRÁFICOS ===
  // ===================================

  /** Processa e atualiza os gráficos Donut e Horizontal */
  atualizarGraficosAgrupados(dados: Ocorrencia[]): void {
    const contagemPorTipo = new Map<string, number>();
    for (const oc of dados) {
      const tipo = oc.type || 'Indefinido';
      contagemPorTipo.set(tipo, (contagemPorTipo.get(tipo) || 0) + 1);
    }

    const contagemOrdenada = Array.from(contagemPorTipo.entries()).sort(
      (a, b) => b[1] - a[1]
    );

    // --- 1. Atualização do Gráfico Donut ---
    const donutLabels = contagemOrdenada.map((item) => item[0]);
    const donutSeries = contagemOrdenada.map((item) => item[1]);

    // Atualizamos as OPÇÕES primeiro
    this.donutChartOptions.series = donutSeries;
    this.donutChartOptions.labels = donutLabels;
    this.donutChartOptions.colors = this.getCoresDonut(donutLabels.length);
    this.donutChartOptions.legend = {
      ...this.donutChartOptions.legend,
      markers: { fillColors: this.getCoresDonut(donutLabels.length) },
    };

    // E DEPOIS atualizamos o gráfico se ele já existir
    if (this.donutChart) {
      this.donutChart.updateOptions({
        series: donutSeries,
        labels: donutLabels,
        colors: this.getCoresDonut(donutLabels.length),
        legend: {
          markers: { fillColors: this.getCoresDonut(donutLabels.length) },
        },
      });
    }

    // --- 2. Atualização do Gráfico Horizontal (Top 5) ---
    const top5Labels = contagemOrdenada.slice(0, 5).map((item) => item[0]);
    const top5Series = contagemOrdenada.slice(0, 5).map((item) => item[1]);

    this.horizontalChartOptions.series = [
      { name: 'Ocorrências', data: top5Series },
    ];
    this.horizontalChartOptions.xaxis.categories = top5Labels;

    if (this.horizontalChart) {
      this.horizontalChart.updateOptions({
        series: [{ name: 'Ocorrências', data: top5Series }],
        xaxis: { categories: top5Labels },
      });
    }
  }

  /** Processa e atualiza o gráfico de Barras (Volume) */
  atualizarGraficoVolumePorPeriodo(dados: Ocorrencia[], periodo: string): void {
    let categories: string[] = [];
    let seriesGrave: number[] = [];
    let seriesSimples: number[] = [];
    const agora = new Date();
    agora.setHours(23, 59, 59, 999);

    if (periodo === '7d') {
      categories = this.getLabelsDiasAnteriores(7);
      [seriesGrave, seriesSimples] = this.agruparDadosPorDia(dados, agora, 7);
    } else if (periodo === '30d') {
      categories = this.getLabelsSemanasAnteriores(5);
      [seriesGrave, seriesSimples] = this.agruparDadosPorSemana(
        dados,
        agora,
        5
      );
    } else if (periodo === '2m') {
      categories = this.getLabelsSemanasAnteriores(9);
      [seriesGrave, seriesSimples] = this.agruparDadosPorSemana(
        dados,
        agora,
        9
      );
    }

    this.barChartOptions.series = [
      { name: 'Erro Grave/Gravíssimo', data: seriesGrave },
      { name: 'Erro Simples/Médio', data: seriesSimples },
    ];
    this.barChartOptions.xaxis.categories = categories;

    if (this.barChart) {
      this.barChart.updateOptions({
        series: this.barChartOptions.series,
        xaxis: this.barChartOptions.xaxis,
      });
    }
  }

  // ===================================
  // === HELPERS DE AGRUPAMENTO ===
  // ===================================
  private getLabelsDiasAnteriores(numDias: number): string[] {
    /* (código mantido) */
    const labels = ['Hoje'];
    for (let i = 1; i < numDias; i++) {
      labels.unshift(i === 1 ? 'Ontem' : `Dia -${i}`);
    }
    return labels;
  }
  private getLabelsSemanasAnteriores(numSemanas: number): string[] {
    /* (código mantido) */
    const labels = ['Esta Semana'];
    for (let i = 1; i < numSemanas; i++) {
      labels.unshift(i === 1 ? 'Semana Passada' : `Semana -${i}`);
    }
    return labels;
  }

  // === CORREÇÃO DE TIPO (2/3) ===
  // Garantimos que a função SEMPRE retorne um boolean
  private isGrave(oc: Ocorrencia): boolean {
    return (
      (oc.severity?.includes('Grave (A)') ?? false) ||
      (oc.severity?.includes('Gravíssima (X)') ?? false)
    );
  }

  private agruparDadosPorDia(
    dados: Ocorrencia[],
    dataFim: Date,
    numDias: number
  ): [number[], number[]] {
    /* (código mantido) */
    const seriesGrave = new Array(numDias).fill(0);
    const seriesSimples = new Array(numDias).fill(0);
    const limitesDias = [];
    for (let i = 0; i < numDias; i++) {
      const data = new Date(dataFim);
      data.setDate(data.getDate() - i);
      data.setHours(0, 0, 0, 0);
      limitesDias.unshift(data);
    }
    for (const oc of dados) {
      const dataOc = new Date(oc.start_ts);
      for (let i = limitesDias.length - 1; i >= 0; i--) {
        if (dataOc >= limitesDias[i]) {
          if (this.isGrave(oc)) seriesGrave[i]++;
          else seriesSimples[i]++;
          break;
        }
      }
    }
    return [seriesGrave, seriesSimples];
  }
  private agruparDadosPorSemana(
    dados: Ocorrencia[],
    dataFim: Date,
    numSemanas: number
  ): [number[], number[]] {
    /* (código mantido) */
    const seriesGrave = new Array(numSemanas).fill(0);
    const seriesSimples = new Array(numSemanas).fill(0);
    const inicioSemanaAtual = new Date(dataFim);
    inicioSemanaAtual.setDate(dataFim.getDate() - dataFim.getDay());
    inicioSemanaAtual.setHours(0, 0, 0, 0);
    const limitesSemanas = [];
    for (let i = 0; i < numSemanas; i++) {
      const data = new Date(inicioSemanaAtual);
      data.setDate(data.getDate() - i * 7);
      limitesSemanas.unshift(data);
    }
    for (const oc of dados) {
      const dataOc = new Date(oc.start_ts);
      for (let i = limitesSemanas.length - 1; i >= 0; i--) {
        if (dataOc >= limitesSemanas[i]) {
          if (this.isGrave(oc)) seriesGrave[i]++;
          else seriesSimples[i]++;
          break;
        }
      }
    }
    return [seriesGrave, seriesSimples];
  }

  // ===========================
  // Detecta tema atual (CORRIGIDO)
  // ===========================
  private getTemaAtual(): 'light' | 'dark' {
    return document.body.classList.contains('light-theme') ? 'light' : 'dark';
  }

  private getCorTexto(): string {
    return this.getTemaAtual() === 'light' ? '#222222' : '#ffffff';
  }

  private getCoresDonut(numCores: number = 7): string[] {
    /* (código mantido) */
    const coresBase =
      this.getTemaAtual() === 'light'
        ? [
            '#FF4B4B',
            '#4B79FF',
            '#FFD700',
            '#00FF00',
            '#FF00FF',
            'gray',
            '#333333',
          ]
        : this.coresPadrao;
    const cores = [];
    for (let i = 0; i < numCores; i++) {
      cores.push(coresBase[i % coresBase.length]);
    }
    return cores;
  }

  // ===========================
  // Carrega ESTRUTURA dos gráficos (CORRIGIDO)
  // ===========================
  carregarEstruturaGraficos() {
    const corTexto = this.getCorTexto();

    // 🔵 Gráfico Vertical (Estrutura VAZIA)
    this.barChartOptions = {
      series: [
        { name: 'Erro Grave/Gravíssimo', data: [] },
        { name: 'Erro Simples/Médio', data: [] },
      ],
      chart: {
        type: 'bar',
        height: 500,
        width: 830,
        animations: { enabled: true },
      },
      plotOptions: {
        bar: { horizontal: false, columnWidth: '55%', borderRadius: 2 },
      },
      dataLabels: { enabled: false },
      stroke: { show: true, width: 2, colors: ['transparent'] },
      xaxis: {
        categories: [],
        labels: { style: { colors: [corTexto], fontSize: '14px' } },
      },
      yaxis: { labels: { style: { colors: [corTexto], fontSize: '14px' } } },
      fill: { opacity: 1, colors: ['#FF4B4B', '#4B79FF'] },
      tooltip: { y: { formatter: (val: number) => val + ' erros' } },
      legend: { position: 'bottom', labels: { colors: corTexto } },
    };

    // 🟣 Gráfico Donut (Estrutura VAZIA)
    this.donutChartOptions = {
      series: [],
      chart: {
        type: 'donut',
        height: 350,
        width: 350,
        animations: { enabled: true },
      },
      labels: [],
      colors: [],
      dataLabels: { enabled: false, style: { colors: [corTexto] } },
      legend: {
        position: 'bottom',
        labels: { colors: corTexto },
        markers: { fillColors: [] },
      },
      responsive: [
        {
          breakpoint: 480,
          options: { chart: { width: 400 }, legend: { position: 'bottom' } },
        },
      ],
    };

    // 🟢 Gráfico Horizontal (Estrutura VAZIA)
    this.horizontalChartOptions = {
      series: [{ name: 'Ocorrências', data: [] }],
      chart: {
        type: 'bar',
        height: 200,
        width: 600,
        animations: { enabled: true },
      },
      plotOptions: { bar: { horizontal: true } },
      dataLabels: { enabled: false },
      xaxis: {
        categories: [],
        labels: { style: { colors: [corTexto], fontSize: '13px' } },
      },
      yaxis: { labels: { style: { colors: [corTexto], fontSize: '13px' } } },
      fill: { opacity: 1, colors: ['#4B79FF'] },
      legend: {
        position: 'right',
        labels: { colors: corTexto },
        markers: { width: 12, height: 12, fillColors: ['#4B79FF'] },
      },
    };
  }

  // ===========================
  // Atualiza cores quando o tema muda (CORRIGIDO)
  // ===========================
  atualizarCoresTema() {
    const corTexto = this.getCorTexto();
    // === CORREÇÃO DE TIPO (3/3) ===
    // Acessa 'labels' de forma segura, pois 'donutChartOptions' já foi inicializado
    const donutLabels = (this.donutChartOptions.labels as string[]) || [];
    const coresDonut = this.getCoresDonut(donutLabels.length);

    if (this.barChart) {
      this.barChart.updateOptions({
        xaxis: { labels: { style: { colors: [corTexto] } } },
        yaxis: { labels: { style: { colors: [corTexto] } } },
        legend: { labels: { colors: corTexto } },
      });
    }

    if (this.donutChart) {
      this.donutChart.updateOptions({
        colors: coresDonut,
        dataLabels: {
          style: { colors: Array(coresDonut.length).fill(corTexto) },
        },
        legend: {
          labels: { colors: corTexto },
          markers: { fillColors: coresDonut },
        },
      });
    }

    if (this.horizontalChart) {
      this.horizontalChart.updateOptions({
        xaxis: { labels: { style: { colors: [corTexto] } } },
        yaxis: { labels: { style: { colors: [corTexto] } } },
        legend: { labels: { colors: corTexto } },
      });
    }
  }

  // =V=========================
  // Aplica filtros (ORQUESTRADOR PRINCIPAL)
  // ===========================
  aplicarFiltros() {
    // --- 1. Filtra por PERÍODO ---
    const agora = new Date();
    let dataLimite = new Date(agora);

    switch (this.periodoSelecionado) {
      case '7d':
        dataLimite.setDate(agora.getDate() - 7);
        break;
      case '30d':
        dataLimite.setDate(agora.getDate() - 30);
        break;
      case '2m':
        dataLimite.setMonth(agora.getMonth() - 2);
        break;
    }

    const dadosFiltradosPorPeriodo = this.dadosCarregados.filter((oc) => {
      return new Date(oc.start_ts) >= dataLimite;
    });

    // --- 2. Atualiza o Gráfico de Barras (Volume) ---
    this.atualizarGraficoVolumePorPeriodo(
      dadosFiltradosPorPeriodo,
      this.periodoSelecionado
    );

    // --- 3. Filtra por TIPO (Grave/Simples) ---
    let dadosFiltradosPorTipo = dadosFiltradosPorPeriodo;
    switch (this.tipoErroSelecionado) {
      case 'grave':
        dadosFiltradosPorTipo = dadosFiltradosPorPeriodo.filter((oc) =>
          this.isGrave(oc)
        );
        break;
      case 'simples':
        dadosFiltradosPorTipo = dadosFiltradosPorPeriodo.filter(
          (oc) => !this.isGrave(oc)
        );
        break;
      case 'todos':
      default:
        break;
    }

    // --- 4. Atualiza os Gráficos Agrupados (Donut e Horizontal) ---
    this.atualizarGraficosAgrupados(dadosFiltradosPorTipo);
  }
}
