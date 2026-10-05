"""IS ANYTHING ON THE LAB STAGE? Reads one real frame off the /ws socket.

There is no GET for the roster — it exists only in the streamed frames, so a
curl cannot answer this and `/state` is a 404 whose body parses to "no ducks"
if you are careless. Usage: python stage_check.py [host:port]
"""
import asyncio, json, sys
import websockets

addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8788"

async def main():
    async with websockets.connect(f"ws://{addr}/ws", open_timeout=8) as ws:
        for _ in range(40):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=8))
            ducks = msg.get("ducks")
            if ducks is None:
                continue
            print(f"{len(ducks)} body(ies) on the {addr} stage")
            for d in ducks:
                who = d.get("id") or d.get("name")
                print(f"   {who}  robot={d.get('robot')}  "
                      f"label={d.get('label') or d.get('title')}")
            return
        print("no frame carrying a roster in 40 messages")

asyncio.run(main())
