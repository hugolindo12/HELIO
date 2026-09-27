# Python – Cálculo de desconto

Erro comum: calcular apenas o valor do desconto em vez do preço final.

Incorreto:
```python
return total * percent / 100
```

Correto (preço após desconto):
```python
return total * (1 - percent / 100)
```

Exemplo: total=60, percent=10 → resultado esperado = 54.0
