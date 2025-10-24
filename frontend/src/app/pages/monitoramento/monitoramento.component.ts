// frontend/src/app/pages/monitoramento/monitoramento.component.ts
// (Versão Completa e Corrigida)

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
import { CommonModule, DatePipe } from '@angular/common'; // Importar CommonModule e DatePipe

// === NOVOS IMPORTS PARA API ===
import { OcorrenciaService } from '../../services/ocorrencia.service';
import { Ocorrencia } from '../../models/ocorrencia'; // Verifique o caminho

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

  // === Sua lista de vídeos de simulação (Mantida) ===
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

  // === Variáveis que serão preenchidas pela API ===
  totalOcorrencias = 0;
  falhasGraves = 0; // Será calculado via API
  ultimaFalha: string | null = null;
  alertas: {
    hora: string;
    mensagem: string;
    tipo: string; // Vai receber "Média (B)", "Grave (A)", "Info (S)" etc.
    animacao: string;
    origem: string;
  }[] = [];
  // ===============================================

  // === Suas variáveis de simulação (Mantidas) ===
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
    // Configuração do Gráfico (Mantida)
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

  // === ngOnInit ATUALIZADO ===
  ngOnInit() {
    // 1. Adiciona o alerta inicial
    this.alertas = [
      {
        hora: new Date().toLocaleTimeString(),
        mensagem: 'Sistema iniciado com sucesso!',
        tipo: 'Info (S)', // Padrão para Sistema
        animacao: 'aparecer',
        origem: 'Sistema',
      },
    ];

    // 2. Chama a nova função para carregar dados da API
    this.carregarDadosIniciais();

    // 3. Inicia o loop do gráfico (lógica original)
    this.startLoop();
  }

  // === NOVA FUNÇÃO PARA CARREGAR DADOS DA API (ALINHADA COM A CARTILHA) ===
  carregarDadosIniciais() {
    console.log('MonitoramentoComponent: Buscando dados iniciais da API...');
    this.ocorrenciaService.getOcorrencias().subscribe(
      (data: Ocorrencia[]) => {
        console.log(`Dados recebidos: ${data.length} ocorrências.`);

        if (data.length > 0) {
          // 1. Atualiza o total
          this.totalOcorrencias = data.length;

          // 2. CORREÇÃO: Conta falhas graves baseado na Cartilha
          // O campo 'severity' DEVE conter "Grave (A)" ou "Gravíssima (X)"
          this.falhasGraves = data.filter(
            (oc) =>
              oc.severity?.includes('Grave (A)') ||
              oc.severity?.includes('Gravíssima (X)')
          ).length;

          // 3. Pega a última falha (baseado no tipo, ex: "Ruído / chiado")
          const maisRecente = data[0]; // API já ordena por ID desc
          this.ultimaFalha = maisRecente.type || 'N/A';

          // 4. Mapeia os dados da API para o formato dos Alertas
          const alertasDaApi = data.map((oc: Ocorrencia) => {
            return {
              hora: this.datePipe.transform(oc.start_ts, 'HH:mm:ss') || 'N/A',
              mensagem: oc.type || 'Ocorrência', // Ex: "Ruído / chiado"
              tipo: oc.severity || 'Info (S)', // Ex: "Média (B)"
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
          tipo: 'Erro (E)', // Padrão para Erro
          animacao: 'aparecer',
          origem: 'Sistema',
        });
      }
    );
  }

  // === NOVAS FUNÇÕES HELPER PARA O BADGE (HTML) ===

  /** Extrai a letra de dentro do parêntese, ex: "Média (B)" -> "b" */
  getBadgeClass(tipo: string): string {
    const match = tipo.match(/\(([^)]+)\)/); // Pega o conteúdo de ( )
    if (match) {
      return match[1].toLowerCase(); // Retorna 'a', 'b', 'c', 'x'
    }
    return 'info'; // Padrão (para "Info (S)" ou "Erro (E)")
  }

  /** Extrai a letra, ex: "Média (B)" -> "B" */
  getBadgeLetra(tipo: string): string {
    const match = tipo.match(/\(([^)]+)\)/);
    if (match) {
      return match[1].toUpperCase(); // Retorna 'A', 'B', 'C', 'X'
    }
    return 'S'; // Padrão (Sistema)
  }

  /** Extrai o texto, ex: "Média (B)" -> "Média" */
  getBadgeTexto(tipo: string): string {
    return tipo.split('(')[0].trim(); // Pega "Média", "Info", "Erro"
  }

  // === FIM DAS NOVAS FUNÇÕES ===

  ngOnDestroy() {
    clearInterval(this.loopInterval);
  }

  // ======================================================
  // === INÍCIO DA LÓGICA DE SIMULAÇÃO ORIGINAL (MANTIDA) ===
  // ======================================================

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

    // NOTA: Esta lógica de simulação agora SOMA aos dados da API.
    // Isso é temporário até a Fase 4 (WebSocket).
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

    // Adiciona alerta (Simulado)
    const mensagens: Record<string, string> = {
      A: 'Repórter Parado',
      B: 'Problema de Áudio',
      C: 'Fade detectado',
      X: 'Freeze detectado',
      P: 'Placa de captura selecionada',
    };

    // Formata o alerta simulado para bater com o padrão da Cartilha
    let severidadeSimulada = 'Info (S)';
    if (video.tipo === 'A') severidadeSimulada = 'Grave (A)';
    if (video.tipo === 'X') severidadeSimulada = 'Gravíssima (X)';
    if (video.tipo === 'B') severidadeSimulada = 'Média (B)';
    if (video.tipo === 'C') severidadeSimulada = 'Leve (C)';

    this.alertas.unshift({
      hora: new Date().toLocaleTimeString(),
      mensagem: mensagens[video.tipo],
      tipo: severidadeSimulada, // Usa o formato da cartilha
      animacao: 'aparecer',
      origem: 'Simulação (JN)', // Indica que é simulação
    });
    this.alertas = this.alertas.slice(0, 4); // Mantém apenas os 4 últimos
    setTimeout(() => this.alertas.forEach((a) => (a.animacao = '')), 1000);
    if (this.alertas.length > 6) this.alertas.pop();

    // Define intensidade do gráfico (Simulado)
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
