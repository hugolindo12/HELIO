"""
Tarefas de código VERIFICADAS para o HEILO Faísca Code: "pedido → código que funciona".

Cada tarefa tem: nome(s) em português e inglês, docstring (o pedido), corpo (a solução)
e testes. Só entra no treino o exemplo cujo código PASSA nos testes, executados de verdade.

Nenhuma tarefa repete as 20 funções da avaliação (seed_code.PROBLEMAS), nem as "irmãs"
diretas delas (ímpar × par, minúsculas × maiúsculas, último × primeiro…): a prova continua
medindo se o modelo generaliza, não se decorou.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (nomes, parametros, docstrings [pt, en], corpo, testes com {f} = nome da função)
T = List[Tuple[List[str], str, List[str], str, str]]

TAREFAS: T = [
    (["produto", "multiplicar"], "a, b", ["Retorna o produto de a e b.", "Return a times b."],
     "return a * b", "assert {f}(3, 4) == 12\nassert {f}(-2, 5) == -10"),
    (["subtrair", "diferenca"], "a, b", ["Retorna a menos b.", "Return a minus b."],
     "return a - b", "assert {f}(10, 3) == 7"),
    (["dividir", "divide"], "a, b", ["Retorna a divisão de a por b.", "Return a divided by b."],
     "return a / b", "assert {f}(10, 4) == 2.5"),
    (["resto", "remainder"], "a, b", ["Retorna o resto da divisão de a por b.", "Return the remainder of a divided by b."],
     "return a % b", "assert {f}(10, 3) == 1"),
    (["potencia", "power"], "base, expoente", ["Retorna base elevado a expoente.", "Return base raised to expoente."],
     "return base ** expoente", "assert {f}(2, 10) == 1024"),
    (["quadrado", "square"], "n", ["Retorna n ao quadrado.", "Return the square of n."],
     "return n * n", "assert {f}(7) == 49"),
    (["cubo", "cube"], "n", ["Retorna n ao cubo.", "Return the cube of n."],
     "return n ** 3", "assert {f}(3) == 27"),
    (["metade", "half"], "n", ["Retorna a metade de n.", "Return half of n."],
     "return n / 2", "assert {f}(9) == 4.5"),
    (["triplo", "triple"], "n", ["Retorna o triplo de n.", "Return three times n."],
     "return n * 3", "assert {f}(4) == 12"),
    (["eh_positivo", "is_positive"], "n", ["Retorna True se n for maior que zero.", "Return True if n is greater than zero."],
     "return n > 0", "assert {f}(3)\nassert not {f}(0)\nassert not {f}(-1)"),
    (["eh_negativo", "is_negative"], "n", ["Retorna True se n for menor que zero.", "Return True if n is negative."],
     "return n < 0", "assert {f}(-3)\nassert not {f}(2)"),
    (["divisivel", "is_divisible"], "a, b", ["Retorna True se a for divisível por b.", "Return True if a is divisible by b."],
     "return a % b == 0", "assert {f}(10, 5)\nassert not {f}(10, 3)"),
    (["valor_absoluto", "absolute"], "n", ["Retorna o valor absoluto de n.", "Return the absolute value of n."],
     "return n if n >= 0 else -n", "assert {f}(-5) == 5\nassert {f}(3) == 3"),
    (["maximo_de_tres", "max_of_three"], "a, b, c", ["Retorna o maior entre a, b e c.", "Return the largest of a, b and c."],
     "return max(a, b, c)", "assert {f}(1, 9, 4) == 9"),
    (["contar_positivos", "count_positive"], "numeros", ["Conta quantos números da lista são positivos.", "Count how many numbers in the list are positive."],
     "return sum(1 for n in numeros if n > 0)", "assert {f}([1, -2, 3, 0]) == 2"),
    (["contar_negativos", "count_negative"], "numeros", ["Conta quantos números da lista são negativos.", "Count the negative numbers in the list."],
     "return sum(1 for n in numeros if n < 0)", "assert {f}([1, -2, -3]) == 2"),
    (["filtrar_positivos", "only_positive"], "numeros", ["Retorna só os números positivos da lista.", "Return only the positive numbers."],
     "return [n for n in numeros if n > 0]", "assert {f}([-1, 2, 0, 5]) == [2, 5]"),
    (["somar_um", "add_one"], "numeros", ["Retorna a lista com 1 somado a cada número.", "Return the list with 1 added to each number."],
     "return [n + 1 for n in numeros]", "assert {f}([1, 5]) == [2, 6]"),
    (["segundo", "second"], "itens", ["Retorna o segundo elemento da lista.", "Return the second element of the list."],
     "return itens[1]", "assert {f}(['a', 'b', 'c']) == 'b'"),
    (["tamanho", "length"], "itens", ["Retorna quantos elementos a lista tem.", "Return how many elements the list has."],
     "return len(itens)", "assert {f}([1, 2, 3]) == 3"),
    (["sem_repetidos", "unique"], "itens", ["Retorna os itens sem repetição, na ordem original.", "Return the items without duplicates, keeping order."],
     "return list(dict.fromkeys(itens))", "assert {f}([3, 1, 3, 2, 1]) == [3, 1, 2]"),
    (["inverter_lista", "reverse_list"], "itens", ["Retorna a lista de trás para frente.", "Return the list reversed."],
     "return itens[::-1]", "assert {f}([1, 2, 3]) == [3, 2, 1]"),
    (["ordenar", "sort_numbers"], "numeros", ["Retorna os números em ordem crescente.", "Return the numbers sorted in ascending order."],
     "return sorted(numeros)", "assert {f}([3, 1, 2]) == [1, 2, 3]"),
    (["ordenar_decrescente", "sort_desc"], "numeros", ["Retorna os números em ordem decrescente.", "Return the numbers in descending order."],
     "return sorted(numeros, reverse=True)", "assert {f}([3, 1, 2]) == [3, 2, 1]"),
    (["contem", "contains"], "itens, valor", ["Retorna True se o valor estiver na lista.", "Return True if value is in the list."],
     "return valor in itens", "assert {f}([1, 2], 2)\nassert not {f}([1, 2], 5)"),
    (["contar_ocorrencias", "count_occurrences"], "itens, valor", ["Conta quantas vezes o valor aparece na lista.", "Count how many times value appears in the list."],
     "return itens.count(valor)", "assert {f}([1, 2, 1, 1], 1) == 3"),
    (["indice", "index_of"], "itens, valor", ["Retorna a posição do valor na lista, ou -1 se não existir.", "Return the index of value in the list, or -1 if missing."],
     "return itens.index(valor) if valor in itens else -1", "assert {f}([5, 6, 7], 7) == 2\nassert {f}([5], 9) == -1"),
    (["juntar_listas", "concat_lists"], "a, b", ["Retorna uma lista com os itens de a seguidos dos de b.", "Return the items of a followed by the items of b."],
     "return a + b", "assert {f}([1], [2, 3]) == [1, 2, 3]"),
    (["primeiros", "take"], "itens, n", ["Retorna os n primeiros itens da lista.", "Return the first n items of the list."],
     "return itens[:n]", "assert {f}([1, 2, 3, 4], 2) == [1, 2]"),
    (["ultimos", "take_last"], "itens, n", ["Retorna os n últimos itens da lista.", "Return the last n items of the list."],
     "return itens[-n:]", "assert {f}([1, 2, 3, 4], 2) == [3, 4]"),
    (["amplitude", "value_range"], "numeros", ["Retorna a diferença entre o maior e o menor número.", "Return the difference between the largest and smallest number."],
     "return max(numeros) - min(numeros)", "assert {f}([4, 9, 1]) == 8"),
    (["tamanho_texto", "text_length"], "texto", ["Retorna quantos caracteres o texto tem.", "Return the number of characters in text."],
     "return len(texto)", "assert {f}('abc') == 3"),
    (["primeira_maiuscula", "capitalize_first"], "texto", ["Retorna o texto com a primeira letra maiúscula.", "Return the text with the first letter capitalized."],
     "return texto[:1].upper() + texto[1:]", "assert {f}('hugo') == 'Hugo'"),
    (["sem_espacos", "strip_spaces"], "texto", ["Remove os espaços do começo e do fim do texto.", "Remove spaces from the start and end of text."],
     "return texto.strip()", "assert {f}('  oi  ') == 'oi'"),
    (["palavras", "split_words"], "texto", ["Retorna a lista de palavras do texto.", "Return the list of words in text."],
     "return texto.split()", "assert {f}('um dois tres') == ['um', 'dois', 'tres']"),
    (["primeira_palavra", "first_word"], "texto", ["Retorna a primeira palavra do texto.", "Return the first word of text."],
     "return texto.split()[0]", "assert {f}('bom dia pessoal') == 'bom'"),
    (["ultima_palavra", "last_word"], "texto", ["Retorna a última palavra do texto.", "Return the last word of text."],
     "return texto.split()[-1]", "assert {f}('bom dia pessoal') == 'pessoal'"),
    (["eh_palindromo", "is_palindrome"], "texto", ["Retorna True se o texto for igual de trás para frente.", "Return True if the text reads the same backwards."],
     "return texto == texto[::-1]", "assert {f}('arara')\nassert not {f}('casa')"),
    (["trocar_espacos", "spaces_to_underscore"], "texto", ["Troca os espaços do texto por _.", "Replace spaces in text with underscores."],
     "return texto.replace(' ', '_')", "assert {f}('a b c') == 'a_b_c'"),
    (["repetir", "repeat_text"], "texto, n", ["Retorna o texto repetido n vezes.", "Return text repeated n times."],
     "return texto * n", "assert {f}('ab', 3) == 'ababab'"),
    (["juntar_palavras", "join_words"], "palavras", ["Junta as palavras da lista com um espaço entre elas.", "Join the words with a single space."],
     "return ' '.join(palavras)", "assert {f}(['bom', 'dia']) == 'bom dia'"),
    (["comeca_com", "starts_with"], "texto, prefixo", ["Retorna True se o texto começar com o prefixo.", "Return True if text starts with prefixo."],
     "return texto.startswith(prefixo)", "assert {f}('heilo', 'he')\nassert not {f}('heilo', 'lo')"),
    (["termina_com", "ends_with"], "texto, sufixo", ["Retorna True se o texto terminar com o sufixo.", "Return True if text ends with sufixo."],
     "return texto.endswith(sufixo)", "assert {f}('arquivo.txt', '.txt')"),
    (["iniciais", "initials"], "nome", ["Retorna as iniciais do nome, em maiúsculas.", "Return the uppercase initials of the name."],
     "return ''.join(p[0].upper() for p in nome.split())", "assert {f}('hugo lopes') == 'HL'"),
    (["saudacao", "greet"], "nome", ["Retorna a frase 'Olá, <nome>!'.", "Return the string 'Olá, <nome>!'."],
     "return f'Olá, {{nome}}!'", "assert {f}('Hugo') == 'Olá, Hugo!'"),
    (["celsius_para_fahrenheit", "c_to_f"], "c", ["Converte graus Celsius para Fahrenheit.", "Convert Celsius to Fahrenheit."],
     "return c * 9 / 5 + 32", "assert {f}(100) == 212\nassert {f}(0) == 32"),
    (["fahrenheit_para_celsius", "f_to_c"], "f", ["Converte graus Fahrenheit para Celsius.", "Convert Fahrenheit to Celsius."],
     "return (f - 32) * 5 / 9", "assert {f}(212) == 100"),
    (["pol_para_mm", "inch_to_mm"], "pol", ["Converte polegadas para milímetros (1 pol = 25.4 mm).", "Convert inches to millimeters (1 in = 25.4 mm)."],
     "return pol * 25.4", "assert abs({f}(2) - 50.8) < 1e-9"),
    (["metros_para_cm", "m_to_cm"], "m", ["Converte metros para centímetros.", "Convert meters to centimeters."],
     "return m * 100", "assert {f}(1.5) == 150"),
    (["horas_para_minutos", "hours_to_minutes"], "h", ["Converte horas para minutos.", "Convert hours to minutes."],
     "return h * 60", "assert {f}(2) == 120"),
    (["segundos_para_minutos", "seconds_to_minutes"], "s", ["Converte segundos para minutos.", "Convert seconds to minutes."],
     "return s / 60", "assert {f}(90) == 1.5"),
    (["porcentagem", "percent_of"], "valor, pct", ["Retorna pct por cento de valor.", "Return pct percent of valor."],
     "return valor * pct / 100", "assert {f}(200, 15) == 30"),
    (["aplicar_desconto", "apply_discount"], "preco, pct", ["Retorna o preço com desconto de pct por cento.", "Return the price after a pct percent discount."],
     "return preco * (1 - pct / 100)", "assert {f}(200, 10) == 180"),
    (["area_retangulo", "rectangle_area"], "base, altura", ["Retorna a área do retângulo.", "Return the area of the rectangle."],
     "return base * altura", "assert {f}(3, 4) == 12"),
    (["perimetro_retangulo", "rectangle_perimeter"], "base, altura", ["Retorna o perímetro do retângulo.", "Return the perimeter of the rectangle."],
     "return 2 * (base + altura)", "assert {f}(3, 4) == 14"),
    (["area_circulo", "circle_area"], "raio", ["Retorna a área do círculo de raio dado.", "Return the area of a circle with the given radius."],
     "import math\n    return math.pi * raio ** 2", "assert abs({f}(1) - 3.14159265) < 1e-6"),
    (["area_triangulo", "triangle_area"], "base, altura", ["Retorna a área do triângulo.", "Return the area of the triangle."],
     "return base * altura / 2", "assert {f}(4, 5) == 10"),
    (["fibonacci", "fib"], "n", ["Retorna o n-ésimo número de Fibonacci (fib(0) = 0, fib(1) = 1).", "Return the n-th Fibonacci number (fib(0) = 0, fib(1) = 1)."],
     "a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "assert {f}(0) == 0\nassert {f}(10) == 55"),
    (["eh_primo", "is_prime"], "n", ["Retorna True se n for primo.", "Return True if n is a prime number."],
     "if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True",
     "assert {f}(7)\nassert not {f}(8)\nassert not {f}(1)"),
    (["somar_digitos", "digit_sum"], "n", ["Retorna a soma dos dígitos de n.", "Return the sum of the digits of n."],
     "return sum(int(d) for d in str(n))", "assert {f}(1234) == 10"),
    (["contar_digitos", "count_digits"], "n", ["Retorna quantos dígitos n tem.", "Return how many digits n has."],
     "return len(str(abs(n)))", "assert {f}(12345) == 5"),
    (["tabuada", "times_table"], "n", ["Retorna a tabuada de n de 1 a 10 numa lista.", "Return the multiplication table of n from 1 to 10."],
     "return [n * i for i in range(1, 11)]", "assert {f}(3)[:3] == [3, 6, 9]\nassert len({f}(3)) == 10"),
    (["numeros_ate", "range_list"], "n", ["Retorna a lista de 1 até n.", "Return the list from 1 to n."],
     "return list(range(1, n + 1))", "assert {f}(4) == [1, 2, 3, 4]"),
    (["pares_ate", "evens_up_to"], "n", ["Retorna os números pares de 0 até n.", "Return the even numbers from 0 to n."],
     "return list(range(0, n + 1, 2))", "assert {f}(6) == [0, 2, 4, 6]"),
    (["mdc", "gcd"], "a, b", ["Retorna o máximo divisor comum de a e b.", "Return the greatest common divisor of a and b."],
     "while b:\n        a, b = b, a % b\n    return a", "assert {f}(12, 18) == 6"),
    (["valores", "dict_values"], "d", ["Retorna a lista de valores do dicionário.", "Return the list of values of the dictionary."],
     "return list(d.values())", "assert {f}({{'a': 1, 'b': 2}}) == [1, 2]"),
    (["pegar_ou_padrao", "get_or_default"], "d, chave, padrao", ["Retorna d[chave], ou padrao se a chave não existir.", "Return d[chave], or padrao if the key is missing."],
     "return d.get(chave, padrao)", "assert {f}({{'a': 1}}, 'b', 0) == 0\nassert {f}({{'a': 1}}, 'a', 0) == 1"),
    (["contar_palavras_dict", "word_frequency"], "texto", ["Retorna um dicionário com quantas vezes cada palavra aparece.", "Return a dict with how many times each word appears."],
     "contagem = {{}}\n    for p in texto.split():\n        contagem[p] = contagem.get(p, 0) + 1\n    return contagem",
     "assert {f}('a b a') == {{'a': 2, 'b': 1}}"),
    (["inverter_dict", "invert_dict"], "d", ["Troca chaves e valores do dicionário.", "Swap the keys and values of the dictionary."],
     "return {{v: k for k, v in d.items()}}", "assert {f}({{'a': 1}}) == {{1: 'a'}}"),
    (["escrever_arquivo", "write_file"], "caminho, texto", ["Grava o texto num arquivo.", "Write text to a file."],
     "with open(caminho, 'w', encoding='utf-8') as f:\n        f.write(texto)",
     "import tempfile, os\np = os.path.join(tempfile.mkdtemp(), 'x.txt')\n{f}(p, 'oi')\nassert open(p, encoding='utf-8').read() == 'oi'"),
    (["contar_linhas", "count_lines"], "caminho", ["Retorna quantas linhas o arquivo de texto tem.", "Return the number of lines in the text file."],
     "with open(caminho, encoding='utf-8') as f:\n        return len(f.read().splitlines())",
     "import tempfile\nt = tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt'); t.write('a\\nb\\nc'); t.close()\nassert {f}(t.name) == 3"),
    (["rpm", "spindle_rpm"], "vc, diametro", ["Calcula a rotação (rpm) a partir da velocidade de corte vc (m/min) e do diâmetro (mm).", "Compute spindle rpm from cutting speed vc (m/min) and diameter (mm)."],
     "import math\n    return vc * 1000 / (math.pi * diametro)", "assert round({f}(100, 10)) == 3183"),
    (["avanco_mm_min", "feed_rate"], "fz, dentes, rpm", ["Calcula o avanço em mm/min: avanço por dente × dentes × rpm.", "Compute feed in mm/min: feed per tooth × teeth × rpm."],
     "return fz * dentes * rpm", "assert {f}(0.05, 4, 2000) == 400"),
    (["tempo_usinagem", "machining_time"], "comprimento, avanco", ["Retorna o tempo em minutos para percorrer o comprimento (mm) no avanço (mm/min).", "Return minutes to travel comprimento (mm) at avanco (mm/min)."],
     "return comprimento / avanco", "assert {f}(300, 150) == 2"),
]


def exemplos(verificar: bool = True) -> Tuple[List[str], Dict]:
    """Monta os exemplos (código completo de cada função) e EXECUTA os testes.
    Retorna (códigos que passaram, relatório)."""
    from heilo.training.seed_code import PROBLEMAS, executar_teste
    proibidos = {n for n, _, _ in PROBLEMAS}
    ok, falhas = [], []
    for nomes, params, docs, corpo, testes in TAREFAS:
        for nome in nomes:
            if nome in proibidos:
                continue
            for doc in docs:
                codigo = f'def {nome}({params}):\n    """{doc}"""\n    {corpo.format()}\n'
                teste = testes.format(f=nome)
                if not verificar or executar_teste(codigo, teste):
                    ok.append(codigo)
                else:
                    falhas.append(nome)
    return ok, {"tarefas": len(TAREFAS), "exemplos": len(ok), "falhas": sorted(set(falhas))}


def preparar_corpus_tarefas(tok, destino, repeticoes: int = 20, peso_fixo: float = 0.3):
    """Corpus pequeno com as tarefas verificadas; peso_fixo = fração dos lotes de treino."""
    import json
    import random
    from pathlib import Path
    from heilo.training import pretrain
    codigos, rel = exemplos()
    rnd = random.Random(0)
    docs = [c for _ in range(repeticoes) for c in rnd.sample(codigos, len(codigos))]
    meta = pretrain.preparar_corpus(iter(docs), tok, Path(destino), max_tokens=10**12, log=lambda s: None)
    meta["peso_fixo"] = peso_fixo
    meta["relatorio"] = rel
    (Path(destino) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
