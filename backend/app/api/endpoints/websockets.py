# backend/app/api/endpoints/websockets.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Importa nosso gerenciador global
from app.websocket_manager import manager

router = APIRouter()

@router.websocket("/ws/ocorrencias")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint WebSocket para receber atualizações de ocorrências em tempo real.
    """
    await manager.connect(websocket)
    try:
        # Mantém a conexão aberta, esperando por mensagens (não esperamos nenhuma do client por enquanto)
        while True:
            # Se precisássemos receber dados do frontend:
            # data = await websocket.receive_text()
            # await manager.send_personal_message(f"Você escreveu: {data}", websocket)

            # Apenas mantém a conexão viva esperando broadcasts do servidor
            await websocket.receive_text() # Espera indefinidamente (ou até desconectar)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("INFO: Cliente WebSocket desconectado.")
    except Exception as e:
         # Trata outros erros inesperados e desconecta
         print(f"ERRO inesperado no WebSocket: {e}")
         manager.disconnect(websocket)


print("INFO: Endpoint WebSocket (/ws/ocorrencias) configurado.")