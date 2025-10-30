# backend/app/api/endpoints/ocorrencias.py
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import csv
import io

from app.db import models, schemas # Nossos modelos e schemas
from app.db.base import get_db     # Nossa dependência de sessão

# Um 'router' é um mini-aplicativo FastAPI para organizar endpoints
router = APIRouter()

# --- PASSO 5: Endpoint de Leitura (READ) ---
@router.get(
    "/ocorrencias",
    response_model=List[schemas.OcorrenciaRead], # Retorna uma LISTA de ocorrências
    summary="Lista todas as ocorrências"
)
def read_ocorrencias(
    db: Session = Depends(get_db), # Injeta a sessão do banco
    skip: int = 0,
    limit: int = 100
):
    """
    Recupera uma lista paginada de ocorrências do banco.
    """
    try:
        ocorrencias = db.query(models.Ocorrencia)\
                         .order_by(models.Ocorrencia.id.desc())\
                         .offset(skip)\
                         .limit(limit)\
                         .all()
        return ocorrencias
    except Exception as e:
        print(f"ERRO ao ler ocorrências: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao buscar ocorrências no banco de dados"
        )


# --- PASSO 6: Endpoint de Escrita (CREATE) ---
@router.post(
    "/ocorrencias",
    response_model=schemas.OcorrenciaRead, # Retorna a ocorrência criada
    status_code=status.HTTP_201_CREATED,  # Retorna "201 Created"
    summary="Cria uma nova ocorrência"
)
def create_ocorrencia(
    ocorrencia: schemas.OcorrenciaCreate, # Valida o corpo do POST
    db: Session = Depends(get_db)         # Injeta a sessão do banco
):
    """
    Cria um novo registro de ocorrência no banco de dados.
    """
    try:
        # Converte o schema Pydantic (ocorrencia) em um modelo SQLAlchemy
        db_ocorrencia = models.Ocorrencia(**ocorrencia.dict())
        
        db.add(db_ocorrencia) # Adiciona à sessão
        db.commit()          # Salva no banco
        db.refresh(db_ocorrencia) # Atualiza o objeto com o ID e created_at
        
        return db_ocorrencia
        
    except Exception as e:
        db.rollback() # Desfaz a transação em caso de erro
        print(f"ERRO ao criar ocorrência: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao salvar ocorrência no banco de dados"
        )

print("INFO: Endpoints de Ocorrências (GET, POST) configurados.")


# --- UPDATE parcial: permite correção humana (descrição/tipo) ---
class OcorrenciaUpdate(BaseModel):
    type: Optional[str] = None
    category: Optional[str] = None
    # severity é derivada de duration_s (cartilha); não editar diretamente
    confidence: Optional[float] = None
    duration_s: Optional[float] = None
    human_description: Optional[str] = None
def _compute_severity(duration_s: Optional[float]) -> Optional[str]:
    try:
        if duration_s is None:
            return None
        d = float(duration_s)
        if d >= 60:
            return "Gravíssima (X)"
        if d >= 10:
            return "Grave (A)"
        if d >= 5:
            return "Média (B)"
        return "Leve (C)"
    except Exception:
        return None


@router.patch(
    "/ocorrencias/{oc_id}",
    response_model=schemas.OcorrenciaRead,
    summary="Atualiza parcialmente uma ocorrência (descrição humana/tipo)"
)
def update_ocorrencia(
    oc_id: int,
    payload: OcorrenciaUpdate,
    db: Session = Depends(get_db)
):
    try:
        db_oc = db.query(models.Ocorrencia).get(oc_id)
        if not db_oc:
            raise HTTPException(status_code=404, detail="Ocorrência não encontrada")

        # Atualiza campos principais se fornecidos
        if payload.type is not None:
            db_oc.type = payload.type
        if payload.category is not None:
            db_oc.category = payload.category
        # confiança é do modelo; não deve ser editada se não for admin. Ignoramos se vier
        # duration pode ser editada (ajuste humano) e atualiza a severidade automáticamente
        if payload.duration_s is not None:
            try:
                db_oc.duration_s = float(payload.duration_s)
            except Exception:
                pass
            sev = _compute_severity(db_oc.duration_s)
            if sev:
                db_oc.severity = sev
        elif (db_oc.severity is None) or (db_oc.severity.lower().startswith('auto')):
            sev = _compute_severity(db_oc.duration_s)
            if sev:
                db_oc.severity = sev

        # Atualiza/insere descrição humana em evidence
        if payload.human_description is not None:
            ev = db_oc.evidence or {}
            ev["human_description"] = payload.human_description
            db_oc.evidence = ev

        db.add(db_oc)
        db.commit()
        db.refresh(db_oc)
        return db_oc
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERRO ao atualizar ocorrência {oc_id}: {e}")
        raise HTTPException(status_code=500, detail="Erro ao atualizar ocorrência")


# --- Exportação CSV ---
@router.get(
    "/ocorrencias/export",
    summary="Exporta ocorrências no formato CSV",
)
def export_ocorrencias_csv(db: Session = Depends(get_db)):
    try:
        ocorrencias = db.query(models.Ocorrencia)\
            .order_by(models.Ocorrencia.id.desc())\
            .all()

        output = io.StringIO(newline='')
        writer = csv.writer(output)
        # Cabeçalho humanizado
        writer.writerow([
            "ID", "Categoria", "Tipo", "Severidade", "Confiança",
            "Início", "Fim", "Duração (s)", "Evidência (path)", "Descrição Humana"
        ])
        for oc in ocorrencias:
            ev = oc.evidence or {}
            writer.writerow([
                oc.id,
                oc.category or '',
                oc.type or '',
                oc.severity or '',
                f"{(oc.confidence or 0):.3f}",
                oc.start_ts.isoformat() if oc.start_ts else '',
                oc.end_ts.isoformat() if oc.end_ts else '',
                f"{(oc.duration_s or 0):.1f}",
                ev.get('path') or ev.get('clip_path') or ev.get('frame') or '',
                ev.get('human_description', ''),
            ])

        csv_bytes = output.getvalue().encode('utf-8-sig')  # BOM para Excel
        headers = {
            'Content-Disposition': 'attachment; filename="ocorrencias.csv"',
            'Content-Type': 'text/csv; charset=utf-8'
        }
        return Response(content=csv_bytes, media_type='text/csv', headers=headers)
    except Exception as e:
        print(f"ERRO ao exportar CSV: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerar CSV")