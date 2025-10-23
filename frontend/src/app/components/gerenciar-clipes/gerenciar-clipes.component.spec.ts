import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GerenciarClipesComponent } from './gerenciar-clipes.component';

describe('GerenciarClipesComponent', () => {
  let component: GerenciarClipesComponent;
  let fixture: ComponentFixture<GerenciarClipesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GerenciarClipesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GerenciarClipesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
