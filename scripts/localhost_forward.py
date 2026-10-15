#!/usr/bin/env python3
"""Forward localhost:<port> to <host>:<port> (default 8000 -> backend:8000).

Used by the docker-compose `worker` container: many cron-driven scripts
(maps/*.py, forecast/DailyForecast*.py) call the API at http://localhost:8000,
but the API runs in the `backend` container. The target host is resolved per
connection so a backend restart/recreate is picked up automatically.
"""
import asyncio
import os
import sys

LISTEN_HOST = "127.0.0.1"
PORT = int(os.getenv("SMF_FORWARD_PORT", "8000"))
TARGET_HOST = os.getenv("SMF_FORWARD_TARGET", "backend")


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle(client_reader, client_writer):
    try:
        up_reader, up_writer = await asyncio.open_connection(TARGET_HOST, PORT)
    except OSError as e:
        print(f"forward: cannot reach {TARGET_HOST}:{PORT}: {e}", file=sys.stderr, flush=True)
        client_writer.close()
        return
    await asyncio.gather(pipe(client_reader, up_writer), pipe(up_reader, client_writer))


async def main():
    server = await asyncio.start_server(handle, LISTEN_HOST, PORT)
    async with server:
        await server.serve_forever()


asyncio.run(main())
