# Padrões de Código Limpo e Boas Práticas em Python (HEILO Standard)

Este documento define as diretrizes fundamentais de engenharia de software para o subagente HEILO CODE e desenvolvedores da plataforma.

## 1. Tipagem Estrita (Type Annotations)
- Todas as funções e métodos devem incluir anotações de tipo completas em argumentos e retorno.
- Utilize `Optional[T]` ou `T | None` para valores opcionais.
- Utilize `List[T]`, `Dict[K, V]` ou as versões nativas do Python 3.9+ (`list[T]`, `dict[K, V]`).

Exemplo:
```python
from typing import Optional, List

def calculate_discounted_total(prices: list[float], discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("O percentual de desconto deve estar entre 0 e 100")
    total = sum(prices)
    return total * (1.0 - (discount_percent / 100.0))
```

## 2. Tratamento de Exceções Explicito
- Nunca use `except:` genérico que engula exceções silenciosamente.
- Capture exceções específicas (`ValueError`, `FileNotFoundError`, `KeyError`).
- Forneça mensagens de erro claras com o contexto do problema.

## 3. Evitar Argumentos Padrão Mutáveis
- Nunca use listas ou dicionários como valores padrão de argumentos (`def foo(items=[])`).
- Use `None` e inicialize dentro da função:
```python
def process_items(items: Optional[list[str]] = None) -> list[str]:
    if items is None:
        items = []
    return [i.strip() for i in items]
```

## 4. Estruturas Imutáveis e Dataclasses
- Prefira `@dataclass` com `frozen=True` para objetos de valor (Value Objects).
- Mantenha funções puras e isoladas sempre que possível, facilitando testes unitários sem efeitos colaterais.

## 5. Gerenciadores de Contexto (Context Managers)
- Sempre utilize `with` para abrir arquivos, conexões de banco de dados ou sockets de rede, garantindo o fechamento seguro mesmo em caso de erro.
```python
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
```
