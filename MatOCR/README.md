# Container approach

  1. definir as bibliotecas mínimas a serem adicionadas
  2. disponibilizar a pasta object_detection contida em models para substituir a de mesmo nome dentro do ambiente virtual (venv)
  3. definir dockerfile para:
    3.1 criar ambiente virtual no linux
    3.2 deploy das bibliotecas mínimas definidas no passo 1
    3.3 percorrer path dentro da Lib até object_detection
    3.4 fazer replace da pasta object_detection existente no path da lib pela provisionada em models/research
  4. Criar ambiente virtual
    - python3 -m venv venv
  5. Entrar no ambiente virtual
    - source venv/bin/activate
  6. Caminho dos packages em Linux
    # Ambiente Virtual Python
    - /venv/lib/python3.8/site-packages
    # Distro Linux
    - /usr/local/lib/python3.8/site-packages