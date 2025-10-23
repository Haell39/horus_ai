# backend/app/api/endpoints/ocorrencias.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

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