# Destilação Teacher → Student (preparado, não implementado)

- **Teacher:** HEILO Teacher (hoje é o Qwen2.5-0.5B-Instruct)
- **Student:** HEILO Seed

## O que a HEILO faz hoje: destilação por exemplos

1. O Teacher recebe perguntas e gera respostas (`/gerar`). As respostas vão para `data/teacher/`.
2. As respostas passam por validação automática (`training/validation.py`) e revisão humana (`/revisar`).
3. Só os exemplos aprovados entram no dataset HEILO (`data/datasets/`).
4. O Seed treina com esse dataset, do mesmo jeito que treina com qualquer outro exemplo.

O Seed aprende a partir das **saídas** do Teacher, e só das que foram aprovadas.

## O que destilação NÃO é

- **Não é copiar pesos.** O Seed tem outra arquitetura, outro tokenizer e outro tamanho. Não existe uma operação que "transfira o cérebro" do Qwen para o Seed.
- **Não é aceitar tudo o que o Teacher diz.** O Teacher também erra, e o que ele gera é tratado como não verificado.

## Possível próximo passo

A destilação por logits (*soft targets*: o Student imita a distribuição de probabilidade do Teacher) **não é compatível hoje**. O Seed usa tokenizer em bytes e o Qwen usa BPE, então os vocabulários não se alinham.

Se isso for implementado no futuro, precisa ser um módulo separado aqui em `training/distillation/`. Esse módulo teria que ser explícito, documentado e desligável, e nunca deve ser chamado pelo Core automaticamente.
