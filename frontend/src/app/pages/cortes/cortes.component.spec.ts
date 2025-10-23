import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CortesComponent } from './cortes.component';

describe('CortesComponent', () => {
  let component: CortesComponent;
  let fixture: ComponentFixture<CortesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CortesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CortesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
