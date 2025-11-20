import { Routes } from '@angular/router';
import { MonitoramentoComponent } from './pages/monitoramento/monitoramento.component';
import { CortesComponent } from './pages/cortes/cortes.component';
import { DadosComponent } from './pages/dados/dados.component';
import { ConfiguracoesComponent } from './pages/configuracoes/configuracoes.component';
import { AcessibilidadeComponent } from './pages/acessibilidade/acessibilidade.component';

export const routes: Routes = [
  { path: '', redirectTo: 'monitoramento', pathMatch: 'full' },
  { path: 'cortes', component: CortesComponent },
  { path: 'configuracoes', component: ConfiguracoesComponent },
  { path: 'acessibilidade', component: AcessibilidadeComponent },
  { path: 'dados', component: DadosComponent },
  { path: 'monitoramento', component: MonitoramentoComponent },
  { path: '**', redirectTo: 'Monitoramento' },
];
