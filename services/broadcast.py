import logging

logger = logging.getLogger(__name__)

connected_clients = set()

async def broadcast_update(data_type: str, data: dict):
    """Broadcast data update to all connected WebSocket clients"""
    message = {
        "type": data_type,
        "data": data
    }
    disconnected = set()
    
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(client)
    
    # Remove disconnected clients
    connected_clients.difference_update(disconnected)

def add_client(client):
    """Add a WebSocket client"""
    connected_clients.add(client)

def remove_client(client):
    """Remove a WebSocket client"""
    connected_clients.discard(client)