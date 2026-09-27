# HEILO Teacher (opcional)

O HEILO Teacher é um **modelo externo e temporário**. Ele funciona como professor: gera exemplos, ajuda na avaliação e acelera a evolução do **HEILO Seed**. **O Teacher não é a HEILO.**

| Campo | Valor |
|---|---|
| Modelo real | `Qwen/Qwen2.5-0.5B-Instruct` (Qwen Team / Alibaba Cloud) |
| Licença | Apache-2.0 ([link](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)) |
| Adaptador | LoRA opcional, em `adapters/heilo_lora/` |
| Finalidade | `training_teacher` |
| Status | `temporary` |

Os metadados oficiais estão em `teacher.json`. Não apague as informações de licença.

## O que o Teacher faz

- Gera respostas para perguntas (`/gerar` ou `python -m heilo.main gerar`). Essas respostas vão para `data/teacher/` como **não verificadas**.
- Serve de referência em `/comparar`. Isso é só uma ferramenta de avaliação: **o Teacher também erra.**
- Só responde no chat se você pedir (`/cerebro teacher` ou `HEILO_TEACHER_IN_CHAT=true`).

## O que o Teacher NÃO faz

- Não controla o Core, a memória nem o conhecimento.
- Não é obrigatório para iniciar a HEILO nem para usá-la offline.
- Não coloca dados no treino sozinho. Tudo o que ele gera passa por validação e aprovação.
- Não "transfere o cérebro" para o Seed. Os pesos do Teacher nunca são copiados.

## Instalar

```
pip install torch transformers peft accelerate
```

Na primeira vez que for usado, o modelo base (~1 GB) é baixado do Hugging Face.

## Desativar

No arquivo `heilo/.env`:

```
HEILO_TEACHER_ENABLED=false
```

## Remover

Apague a pasta `heilo/models/teacher/`. A HEILO continua iniciando, conversando com o Seed e usando memória e conhecimento. As funções que dependem do Teacher respondem: "Esta função requer o HEILO Teacher."

## Trocar de professor

Edite `teacher.json` (`base_model`, `license`, `origin`...). O Core não precisa mudar.
