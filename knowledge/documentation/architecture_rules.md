# Regras de Arquitetura e Segurança da HEILO

Este documento estabelece as diretrizes de governança e princípios operacionais do sistema HEILO.

## 1. Princípio Local-First (Conhecimento Próprio Primeiro)
- Toda consulta ou problema enviado pelo usuário deve primeiro ser avaliado pelo `KnowledgeBrain`.
- Se a base local de conhecimento ou histórico de soluções verificadas (`verified_solutions`) possuir uma resposta com confiança superior ao limiar (`>= 0.35`), o sistema responde imediatamente a partir da memória local.
- Modelos LLM externos (OpenAI, Ollama, etc.) atuam exclusivamente como mecanismo de síntese ou fallback para problemas totalmente inéditos.

## 2. Isolamento Estrito de Workspace
- Nenhum agente tem permissão para ler ou modificar caminhos fora do diretório autorizado do workspace (`WorkspaceManager`).
- Tentativas de path traversal (`..`, caminhos absolutos de sistema como `/etc`, `C:\Windows`, `System32`) são bloqueadas e registradas com exceção de segurança.

## 3. Padrão Checkpoint-Before-Write
- Antes de qualquer mutação destrutiva em disco (`write_file`, `edit_file`), o `CheckpointManager` cria um snapshot prévio do arquivo.
- Se a suíte de testes falhar após a alteração (`HEILO TEST` falha), o sistema executa rollback automático imediatamente para restaurar o estado original e impedir arquivos corrompidos.

## 4. Governança por Permissões (Princípio do Menor Privilégio)
- Operações de leitura (`READ`) são livres.
- Operações de escrita (`WRITE`) e execução (`EXECUTE`) podem exigir aprovação explícita conforme a política de segurança ativa.
- Comandos críticos (`CRITICAL`) sempre exigem confirmação do usuário.
