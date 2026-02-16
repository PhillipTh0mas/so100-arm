import asyncio
import logging

from app.agent import run_agent
from app.webui import serve_webui

logging.basicConfig(level=logging.INFO)


async def agent_loop():
    while True:
        try:
            await run_agent()
        except Exception as e:
            logging.error(e)
            await asyncio.sleep(5)
        except SystemExit as e:
            logging.error(f"Session SystemExit caught ({e.code}); will restart.")
            await asyncio.sleep(5)


async def main():
    await asyncio.gather(
        serve_webui(),
        agent_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
