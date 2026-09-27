# verified_solution: Com base na pesquisa abaixo, aplique a solucao no projeto do workspace.
Pedido o

- **id:** `20260927_013705_863083`
- **created:** 2026-09-27T01:37:05.863122+00:00
- **confidence:** verified
- **project:** demo_project
- **agent:** -
- **tests_passed:** True

## Problem
Com base na pesquisa abaixo, aplique a solucao no projeto do workspace.
Pedido original do usuario: Pesquise a solucao de desconto em Python, aplique no projeto e rode os testes.

--- Pesquisa (HEILO RESEARCH) ---
**HEILO** (via HEILO RESEARCH)

FINAL: Síntese da pesquisa

## Resumo
Para calcular o preço final após um desconto percentual em Python, use:

```python
final = total * (1 - percent / 100)
```

Erro comum: `total * percent / 100` retorna só o valor do desconto, não o preço final.

Exemplo: total=60, percent=10 -> final=54.0

## Fontes
- knowledge/programming/python_discounts.md
- Documentação comum de cálculo comercial

Sources collected:
- [programming\python_discounts.md] (score=0.72) # Python – Cálculo de desconto

Erro comum: calcular apenas o valor do de
- [programming\pytes

## Correction
FINAL: Correção concluída com sucesso.

Problema: a função apply_discount calculava o valor do desconto em vez do preço final.
Causa: return total * percent / 100
Correção: return total * (1 - percent / 100)
Arquivo: main.py
Testes: PASS

## Files
- `main.py`

## Tags
discount, desconto, pytest, python
