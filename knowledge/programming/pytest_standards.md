# Padrões de Testes com Pytest (HEILO Standard)

Este documento define como o subagente HEILO TEST e desenvolvedores devem escrever e validar suítes de testes na plataforma.

## 1. Padrão Arrange-Act-Assert (AAA)
Todo teste unitário deve seguir a estrutura tripartite:
- **Arrange (Preparação):** Configuração de variáveis, mocks e estado inicial.
- **Act (Execução):** Chamada da função ou método a ser testado.
- **Assert (Validação):** Verificação rigorosa do resultado esperado.

```python
def test_calculate_discounted_total_success():
    # Arrange
    prices = [10.0, 20.0, 30.0]
    discount = 10.0
    expected = 54.0

    # Act
    actual = calculate_discounted_total(prices, discount)

    # Assert
    assert abs(actual - expected) < 1e-6
```

## 2. Cobertura de Casos Limite (Edge Cases)
Sempre teste cenários adversos e de fronteira:
1. **Coleções vazias:** lista de preços vazia `[]`.
2. **Valores nulos ou neutros:** desconto de 0% ou 100%.
3. **Casos inválidos:** valores negativos ou percentuais acima de 100%.

```python
import pytest

def test_calculate_discount_invalid_percent_raises():
    with pytest.raises(ValueError, match="entre 0 e 100"):
        calculate_discounted_total([100.0], -5.0)
```

## 3. Testes Parametrizados (`@pytest.mark.parametrize`)
Evite duplicar código de teste para diferentes entradas e saídas. Utilize parametrização:

```python
import pytest

@pytest.mark.parametrize("prices,percent,expected", [
    ([100.0], 0.0, 100.0),
    ([100.0], 20.0, 80.0),
    ([50.0, 50.0], 50.0, 50.0),
    ([10.0, 20.0, 30.0], 10.0, 54.0),
])
def test_discount_matrix(prices, percent, expected):
    assert abs(calculate_discounted_total(prices, percent) - expected) < 1e-6
```

## 4. Uso de Fixtures para Isolamento
Utilize `tmp_path` nativo do pytest para operações de sistema de arquivos e evite compartilhar estado mutável entre testes.
