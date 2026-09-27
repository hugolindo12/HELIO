# Python Asyncio – conceitos básicos

- `async def` define uma corrotina
- `await` suspende até a operação assíncrona completar
- `asyncio.run(main())` é o entrypoint comum
- Use `asyncio.gather` para concorrência

Exemplo:
```python
import asyncio

async def fetch(url):
    await asyncio.sleep(0.1)
    return url

async def main():
    results = await asyncio.gather(fetch("a"), fetch("b"))
    print(results)

asyncio.run(main())
```
