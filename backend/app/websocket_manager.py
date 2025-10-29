# backend/app/websocket_manager.py
from fastapi import WebSocket
from typing import List, Dict
import json

class ConnectionManager:
    def __init__(self):
        # Guarda as conexões ativas
        self.active_connections: List[WebSocket] = []
        print("INFO: ConnectionManager inicializado.")

    async def connect(self, websocket: WebSocket):
        """ Aceita uma nova conexão WebSocket. """
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"INFO: Nova conexão WebSocket estabelecida. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """ Remove uma conexão WebSocket. """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"INFO: Conexão WebSocket fechada. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """ Envia uma mensagem para uma conexão específica. """
        try:
             await websocket.send_text(message)
        except Exception as e:
             print(f"ERRO ao enviar msg pessoal para WS: {e}. Desconectando.")
             self.disconnect(websocket)


    async def broadcast(self, message: str):
        """ Envia uma mensagem para TODAS as conexões ativas. """
        disconnected_sockets = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                # Se falhar ao enviar (ex: navegador fechou), marca para remover
                print(f"ERRO ao enviar broadcast para WS: {e}. Marcando para desconexão.")
                disconnected_sockets.append(connection)

        # Remove conexões que falharam
        for socket in disconnected_sockets:
            self.disconnect(socket)

        # Log apenas se houver conexões ativas restantes
        # if self.active_connections:
        #      print(f"DEBUG: Broadcast enviado para {len(self.active_connections)} conexões.")


    async def broadcast_json(self, data: Dict):
         """ Converte um dicionário para JSON e envia para todos. """
         await self.broadcast(json.dumps(data, default=str)) # default=str para lidar com datetimes

# Cria uma instância única do gerenciador que será usada pela aplicação
manager = ConnectionManager()

print("INFO: Instância global ConnectionManager criada.")