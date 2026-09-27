"""
HEILO Models — componentes de modelo substituíveis.

- seed/     HEILO Seed: modelo próprio, treinado do zero (projeto de longo prazo)
- teacher/  HEILO Teacher: modelo externo temporário [OPCIONAL — pode ser apagado]
- base.py   contrato ModelAdapter que o Core usa
- registry  monta os adaptadores sem que o Core importe nada de modelo
"""
