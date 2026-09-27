# HEILO Cloud (preparado)

Esta é a infraestrutura futura. **Ainda não está implementada.**

```
HEILO Local → HEILO Cloud API → Model Server | Memory | Knowledge | RAG | Database
```

- **Modelos:** entram no Core como mais um `ModelAdapter` (`cloud/__init__.py`, `CloudModelAdapter`).
- **Memory e Knowledge:** hoje são locais (`memory/manager.py` e `knowledge/manager.py`). Na Cloud, basta manter a mesma interface com outro armazenamento.
- **Teacher:** pode futuramente rodar na Cloud, mas continua sendo um componente externo e opcional.
- A HEILO continua funcionando offline, sem a Cloud.
