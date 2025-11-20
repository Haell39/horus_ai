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
  tooltip?: ApexTooltip;
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
  tooltip?: ApexTooltip;
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
  @ViewChild('hourChart') hourChart!: ChartComponent;
  @ViewChild('kpiSparkline') kpiSparkline!: ChartComponent;
  @ViewChild('kpiTotalSparkline') kpiTotalSparkline!: ChartComponent;
  @ViewChild('kpiDonutChart') kpiDonutChart!: ChartComponent;

  // === CORREÇÃO DE TIPO (1/3) ===
  // Removemos o 'Partial' e usamos '!' para dizer ao TS que vamos inicializar
  public barChartOptions!: BarChartOptions;
  public donutChartOptions!: DonutChartOptions;
  public horizontalChartOptions!: HorizontalBarOptions;
  public hourChartOptions!: BarChartOptions;
  // KPI card state (percent graves + sparkline)
  public kpiPercentGraves: number = 0;
  public kpiDeltaPercent: number | null = null;
  public kpiSparklineOptions: any;
  public kpiSparklineSeries: any[] = [];
  // KPI Total
  public kpiTotalCount: number = 0;
  public kpiTotalDelta: number | null = null;
  public kpiTotalSparklineOptions: any;
  public kpiTotalSparklineSeries: any[] = [];
  // KPI Donut (Graves vs Outros) — ligado ao filtro de período
  public kpiDonutOptions!: DonutChartOptions;
  public kpiDonutSeries: number[] = [];

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
  // filtered view (after period + tipo filters) — exposed for template/debug
  public filteredDados: Ocorrencia[] = [];
  // last computed dataLimite (ISO) for debugging
  public lastDataLimite: string | null = null;
  // debug: earliest and latest start_ts in the current loaded dataset
  public minStartTs: string | null = null;
  public maxStartTs: string | null = null;

  constructor(
    private ocorrenciaService: OcorrenciaService,
    private datePipe: DatePipe
  ) {}

  ngOnInit(): void {
    // Inicialização
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
    // request a larger page so the Dados page can filter/aggregate on the full set
    this.ocorrencias$ = this.ocorrenciaService.getOcorrencias(1000).pipe(
      tap((data) => {
        this.dadosCarregados = data;
        // fetch the true total from the backend (uncapped count)
        this.ocorrenciaService.getTotalOcorrencias().subscribe({
          next: (res) => {
            try {
              this.kpiTotalCount = res.count ?? data.length;
            } catch (e) {
              this.kpiTotalCount = data.length;
            }
          },
          error: () => {
            this.kpiTotalCount = data.length;
          },
        });
        // initialize filtered view and apply filters to update KPIs/charts
        this.filteredDados = data;
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
  atualizarGraficoVolumePorPeriodo(
    dados: Ocorrencia[],
    periodo: string,
    dataFim: Date
  ): void {
    let categories: string[] = [];
    let seriesGrave: number[] = [];
    let seriesSimples: number[] = [];
    const agora = new Date(dataFim);
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
      tooltip: {
        y: { formatter: (val: number) => val + ' erros' },
        theme: 'dark' as any,
      },
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
        labels: { colors: corTexto, style: { fontSize: '14px' } },
        markers: { width: 14, height: 14, fillColors: [] },
      },
      tooltip: { theme: 'dark' as any },
      responsive: [
        {
          breakpoint: 480,
          options: { chart: { width: 400 }, legend: { position: 'bottom' } },
        },
      ],
    } as any;

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
      tooltip: { theme: 'dark' as any },
    } as any;

    // 🕒 Gráfico: Ocorrências por Hora do Dia (Estrutura VAZIA)
    this.hourChartOptions = {
      series: [{ name: 'Ocorrências', data: [] }],
      chart: {
        type: 'bar',
        height: 260,
        width: 830,
        animations: { enabled: true },
      },
      plotOptions: {
        bar: { horizontal: false, columnWidth: '60%', borderRadius: 4 },
      },
      dataLabels: { enabled: false },
      stroke: { show: true, width: 1, colors: ['transparent'] },
      xaxis: { categories: Array.from({ length: 24 }, (_, i) => `${i}h`) },
      yaxis: {
        labels: { style: { colors: [this.getCorTexto()], fontSize: '13px' } },
      },
      fill: { opacity: 1, colors: ['#1E88E5'] },
      tooltip: { y: { formatter: (val: number) => `${val} ocorrências` } },
      legend: { show: false },
    };

    // KPI sparkline (compact)
    this.kpiSparklineOptions = {
      series: [],
      chart: { type: 'area', height: 60, sparkline: { enabled: true } },
      stroke: { curve: 'smooth', width: 2, colors: ['#FF4B4B'] },
      markers: { size: 4, colors: ['#FF4B4B'], strokeColors: '#ffffff' },
      fill: { opacity: 0.28, colors: ['#FFCDD2'] },
      tooltip: { enabled: true, theme: 'dark' },
      colors: ['#FF4B4B'],
    };

    // KPI total sparkline (counts)
    this.kpiTotalSparklineOptions = {
      series: [],
      chart: { type: 'area', height: 60, sparkline: { enabled: true } },
      stroke: { curve: 'smooth', width: 2 },
      fill: { opacity: 0.25, colors: ['#1E88E5'] },
      tooltip: { enabled: false, theme: 'dark' },
      colors: ['#1E88E5'],
    };

    // KPI Donut (Graves vs Outros) initial options
    // Note: keep the KPI card height small; make the chart fit the card by using a smaller height
    // and a responsive width. Also set the donut inner size so it looks balanced inside the card.
    this.kpiDonutOptions = {
      series: [0, 0],
      // increase KPI donut so it is visually prominent when stacked vertically
      chart: { type: 'donut', height: 260, width: '100%' },
      labels: ['Graves/Gravíssimas', 'Outros'],
      /* cores com contraste maior para 'Outros' */
      colors: ['#FF4B4B', '#9AA6C1'],
      dataLabels: { enabled: false },
      legend: {
        position: 'bottom',
        labels: { colors: this.getCorTexto(), style: { fontSize: '13px' } },
        markers: { width: 12, height: 12 },
      },
      plotOptions: { pie: { donut: { size: '55%' } } },
      responsive: [
        { breakpoint: 1400, options: { chart: { height: 240 } } },
        { breakpoint: 1200, options: { chart: { height: 220 } } },
        { breakpoint: 900, options: { chart: { height: 200 } } },
        { breakpoint: 768, options: { chart: { height: 180 } } },
        { breakpoint: 480, options: { chart: { height: 140 } } },
      ],
    } as any;
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

    if (this.kpiDonutChart) {
      try {
        this.kpiDonutChart.updateOptions({
          legend: {
            labels: { colors: corTexto },
            markers: {
              width: 14,
              height: 14,
              fillColors: this.kpiDonutOptions.colors,
            },
          },
        });
      } catch (e) {
        // ignore if not ready
      }
    }
  }

  // =V=========================
  // Aplica filtros (ORQUESTRADOR PRINCIPAL)
  // ===========================
  aplicarFiltros() {
    // Recompute min/max timestamps from the loaded dataset first (used to anchor windows)
    try {
      const dates = this.dadosCarregados
        .map((o) => new Date(o.start_ts).getTime())
        .filter((t) => !isNaN(t));
      if (dates.length > 0) {
        this.minStartTs = new Date(Math.min(...dates)).toISOString();
        this.maxStartTs = new Date(Math.max(...dates)).toISOString();
      } else {
        this.minStartTs = null;
        this.maxStartTs = null;
      }
    } catch (e) {
      this.minStartTs = null;
      this.maxStartTs = null;
    }

    // Anchor period windows to the latest available event if present, otherwise use now()
    const referenceNow = this.maxStartTs
      ? new Date(this.maxStartTs)
      : new Date();
    referenceNow.setHours(23, 59, 59, 999);

    // --- 1. Filtra por PERÍODO (anchored at referenceNow) ---
    let dataLimite = new Date(referenceNow);
    switch (this.periodoSelecionado) {
      case '7d':
        dataLimite.setDate(referenceNow.getDate() - 7);
        break;
      case '30d':
        dataLimite.setDate(referenceNow.getDate() - 30);
        break;
      case '2m':
        dataLimite.setMonth(referenceNow.getMonth() - 2);
        break;
    }

    const dadosFiltradosPorPeriodo = this.dadosCarregados.filter((oc) => {
      return new Date(oc.start_ts) >= dataLimite;
    });
    // store last dataLimite for internal use
    this.lastDataLimite = dataLimite.toISOString();

    // --- 2. Atualiza o Gráfico de Barras (Volume) ---
    this.atualizarGraficoVolumePorPeriodo(
      dadosFiltradosPorPeriodo,
      this.periodoSelecionado,
      referenceNow
    );

    // --- KPI Donut: Graves vs Outros (aplica somente filtro de PERÍODO) ---
    try {
      const graves = dadosFiltradosPorPeriodo.filter((oc) =>
        this.isGrave(oc)
      ).length;
      const totalPeriodo = dadosFiltradosPorPeriodo.length;
      const outros = Math.max(0, totalPeriodo - graves);
      this.kpiDonutSeries = [graves, outros];
      this.kpiDonutOptions.series = this.kpiDonutSeries as any;
      this.kpiDonutOptions.labels = ['Graves/Gravíssimas', 'Outros'];
      this.kpiDonutOptions.colors = ['#FF4B4B', '#9AA6C1'];
      if (this.kpiDonutChart) {
        this.kpiDonutChart.updateOptions({
          series: this.kpiDonutSeries,
          labels: this.kpiDonutOptions.labels,
          colors: this.kpiDonutOptions.colors,
        });
      }
    } catch (e) {
      // ignore
    }

    // (hour chart update call will be executed after tipo filtering)
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

    // --- 4. Atualiza o Gráfico de Ocorrências por Hora ---
    this.atualizarGraficoHoras(dadosFiltradosPorTipo);
    // update filteredDados for template
    this.filteredDados = dadosFiltradosPorTipo;
    // --- 4.5 Atualiza o cartão KPI (Percent % Ocorrências Graves) ---
    this.atualizarKpiCard(
      dadosFiltradosPorTipo,
      dataLimite,
      this.periodoSelecionado,
      referenceNow
    );

    // --- 5. Atualiza os Gráficos Agrupados (Donut e Horizontal) ---
    this.atualizarGraficosAgrupados(dadosFiltradosPorTipo);
  }

  /** Atualiza o cartão KPI: principal = % Ocorrências Graves
   *  - dados: já filtrados por período e tipo (tipo usado to compute sparkline buckets)
   *  - dataLimite: início do período (exclusive) -> usado para calcular período anterior
   */
  private atualizarKpiCard(
    dados: Ocorrencia[],
    dataLimite: Date,
    periodo: string,
    agoraRef: Date
  ) {
    try {
      // Current period counts
      const totalAtual = dados.length;
      const gravesAtual = dados.filter((oc) => this.isGrave(oc)).length;
      const percentAtual =
        totalAtual === 0 ? 0 : (gravesAtual / totalAtual) * 100;

      // Determine window length in ms for previous period calculation
      const agora = new Date(agoraRef);
      let windowMs = 7 * 24 * 3600 * 1000; // default 7d
      if (periodo === '7d') windowMs = 7 * 24 * 3600 * 1000;
      else if (periodo === '30d') windowMs = 30 * 24 * 3600 * 1000;
      else if (periodo === '2m') windowMs = 60 * 24 * 3600 * 1000;

      const periodoInicio = dataLimite.getTime();
      const periodoFim = agora.getTime();
      const prevInicio = periodoInicio - (periodoFim - periodoInicio);
      const prevFim = periodoInicio;

      // Previous period filtered using same tipo selection on full dadosCarregados
      const prevWindow = this.dadosCarregados.filter((oc) => {
        const t = new Date(oc.start_ts).getTime();
        return (
          t >= prevStartBoundary(prevInicio) && t < prevEndBoundary(prevFim)
        );
      });

      // Apply same tipo filter to prevWindow
      let prevFiltered = prevWindow;
      if (this.tipoErroSelecionado === 'grave')
        prevFiltered = prevWindow.filter((oc) => this.isGrave(oc));
      else if (this.tipoErroSelecionado === 'simples')
        prevFiltered = prevWindow.filter((oc) => !this.isGrave(oc));

      const totalPrev = prevFiltered.length;
      const gravesPrev = prevFiltered.filter((oc) => this.isGrave(oc)).length;
      const percentPrev = totalPrev === 0 ? 0 : (gravesPrev / totalPrev) * 100;

      // Delta (percentage points change)
      let delta = null;
      if (totalPrev === 0) delta = null;
      else delta = percentAtual - percentPrev;

      this.kpiPercentGraves = Number(percentAtual.toFixed(1));
      this.kpiDeltaPercent = delta === null ? null : Number(delta.toFixed(1));

      // Build sparkline series: compute percent graves per bucket (day/week)
      let buckets: number[] = [];
      if (periodo === '7d') {
        // use already filtered `dados` (period + tipo) to compute percent buckets
        const [g, s] = this.agruparDadosPorDia(dados, agoraRef, 7);
        buckets = g.map((gv, i) => {
          const total = gv + s[i];
          return total === 0 ? 0 : (gv / total) * 100;
        });
      } else {
        // for 30d and 2m use weekly buckets
        const weeks = periodo === '30d' ? 5 : 9;
        // use already filtered `dados` (period + tipo) to compute percent buckets (weekly)
        const [g, s] = this.agruparDadosPorSemana(dados, agoraRef, weeks);
        buckets = g.map((gv, i) => {
          const total = gv + s[i];
          return total === 0 ? 0 : (gv / total) * 100;
        });
      }

      this.kpiSparklineSeries = [
        { name: '% Graves', data: buckets.map((v) => Number(v.toFixed(1))) },
      ];
      // update chart if exists
      if (this.kpiSparkline) {
        try {
          this.kpiSparkline.updateOptions({ series: this.kpiSparklineSeries });
        } catch (e) {
          // ignore if chart not ready
        }
      }
      // Also update Total KPI sparkline and values
      // Build total buckets (counts) matching the same buckets used above
      // Build total buckets (counts) from the already filtered `dados`
      if (periodo === '7d') {
        const [, totals] = this.agruparDadosPorDia(dados, agoraRef, 7);
        this.kpiTotalSparklineSeries = [
          { name: 'Total', data: totals.map((v) => v) },
        ];
      } else {
        const weeks = periodo === '30d' ? 5 : 9;
        const [, totals] = this.agruparDadosPorSemana(dados, agoraRef, weeks);
        this.kpiTotalSparklineSeries = [
          { name: 'Total', data: totals.map((v) => v) },
        ];
      }

      // compute totals for current period (do not overwrite global total from backend)
      // keep backend-provided total in `this.kpiTotalCount` (set in carregarDadosApi)
      // previous period total
      const prevTotal = this.dadosCarregados.filter((oc) => {
        const t = new Date(oc.start_ts).getTime();
        return (
          t >=
            dataLimite.getTime() - (agora.getTime() - dataLimite.getTime()) &&
          t < dataLimite.getTime()
        );
      }).length;
      this.kpiTotalDelta =
        prevTotal === 0 ? null : this.kpiTotalCount - prevTotal;

      if (this.kpiTotalSparkline) {
        try {
          this.kpiTotalSparkline.updateOptions({
            series: this.kpiTotalSparklineSeries,
          });
        } catch (e) {
          // ignore
        }
      }
    } catch (e) {
      console.error('Erro ao atualizar KPI card', e);
    }

    // helper closures to guard invalid epoch boundaries
    function prevStartBoundary(v: number) {
      return isFinite(v) ? v : 0;
    }
    function prevEndBoundary(v: number) {
      return isFinite(v) ? v : Date.now();
    }
  }

  /** Atualiza o gráfico de Ocorrências por Hora do Dia (0-23) */
  private atualizarGraficoHoras(dados: Ocorrencia[]) {
    const counts = new Array(24).fill(0);
    for (const oc of dados) {
      try {
        const d = new Date(oc.start_ts);
        if (isNaN(d.getTime())) continue;
        const h = d.getHours();
        counts[h] = (counts[h] || 0) + 1;
      } catch (e) {
        // ignore malformed dates
      }
    }

    this.hourChartOptions.series = [{ name: 'Ocorrências', data: counts }];
    this.hourChartOptions.xaxis = {
      ...(this.hourChartOptions.xaxis as any),
      categories: Array.from({ length: 24 }, (_, i) => `${i}h`),
    };

    if (this.hourChart) {
      try {
        this.hourChart.updateOptions({
          series: this.hourChartOptions.series,
          xaxis: this.hourChartOptions.xaxis,
        });
      } catch (e) {
        // updateOptions may fail if chart not yet initialized; ignore
      }
    }
  }
}
