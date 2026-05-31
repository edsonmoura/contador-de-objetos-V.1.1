# Contador de Objetos

Ferramenta em Python para praticar visao computacional contando objetos pela camera.

O app usa:

- Kivy para a interface e acesso a camera.
- OpenCV DNN para rodar um modelo YOLO em formato ONNX.
- Python puro para agrupar os objetos detectados por tipo e somar o total.

## Como funciona

1. A camera mostra a imagem ao vivo.
2. Voce toca em **Contar**.
3. O app captura o frame atual.
4. O detector identifica os objetos.
5. A tela mostra:
   - quantidade por tipo de objeto;
   - total geral de objetos;
   - imagem com caixas desenhadas.

Exemplo de saida:

```text
Total: 7

person: 2
bottle: 3
cup: 2
```

## Estrutura

```text
.
+-- main.py                 # App Kivy
+-- object_counter/
|   +-- counter.py          # Agrupa deteccoes e soma objetos
|   +-- detector.py         # Detector YOLO ONNX com OpenCV
|   +-- labels.py           # Labels COCO
+-- scripts/
|   +-- count_image.py      # Testa o contador com uma imagem local
|   +-- download_model.py   # Baixa um modelo YOLOv8n ONNX
+-- .github/workflows/
|   +-- build-android.yml   # Gera APK pelo GitHub Actions
+-- requirements.txt        # Dependencias para testar no computador
+-- requirements-model.txt  # Dependencia para gerar o modelo ONNX
+-- buildozer.spec          # Base para empacotar no Android
```

## Testar no computador

Instale as dependencias:

```powershell
python -m pip install -r requirements.txt
```

Baixe o modelo:

```powershell
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

Se aparecer erro do ONNX no Windows, como `WinError 206` ou `No module named 'onnx.defs'`, a instalacao do ONNX ficou incompleta. Rode:

```powershell
python -m pip uninstall -y onnx
python -m pip install --no-cache-dir --force-reinstall onnx==1.16.2
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

Esse erro e comum com Python instalado pela Microsoft Store porque o caminho de instalacao fica muito longo.

Se o erro continuar mesmo com `onnx==1.16.2`, nao insista no Python da Microsoft Store para gerar ONNX. Use um destes caminhos:

1. Teste localmente com `scripts/count_image.py`, que consegue usar `yolov8n.pt`.
2. Gere o APK pelo GitHub Actions, que roda em Linux e exporta o ONNX fora do seu PC.
3. Instale Python pelo site python.org ou crie um ambiente virtual em caminho curto, como `C:\venvs\contador`.

Rode o app:

```powershell
python main.py
```

Se o computador nao tiver camera, o app deve abrir mostrando um aviso. Nesse caso, teste a deteccao usando uma imagem local ou instale o APK no celular.

No Windows, voce tambem pode abrir:

```text
run_app.bat
```

Ele mantem o terminal aberto caso o app feche com erro.

Para verificar dependencias e modelo:

```powershell
python scripts/check_setup.py
```

## Testar sem camera no desktop

Se o seu computador nao tem camera, use uma imagem salva:

```powershell
python scripts/count_image.py .\minha_foto.jpg
```

Troque `.\minha_foto.jpg` pelo caminho real da foto que voce quer testar.

Para objetos escuros em fundo claro, como graos, pecas pequenas ou pedras sobre uma mesa clara, use:

```powershell
python scripts/count_image.py .\imagem.jpg --mode dark
```

Tambem existem presets:

```powershell
python scripts/count_image.py .\feijao.jpg --mode beans
python scripts/count_image.py .\arroz.jpg --mode rice
python scripts/count_image.py .\parafusos.jpg --mode parts
```

Se algum objeto pequeno nao for contado, reduza a area minima:

```powershell
python scripts/count_image.py .\imagem.jpg --mode dark --min-area 400
```

Para arroz, talvez seja necessario um valor menor:

```powershell
python scripts/count_image.py .\arroz.jpg --mode rice --min-area 20
```

No Windows, esse teste aceita dois modelos:

- `models/yolov8n.onnx`, usado pelo app Kivy/OpenCV.
- `yolov8n.pt`, usado como fallback pelo Ultralytics quando o ONNX ainda nao existe.

O terminal mostrara a contagem por tipo e o total. A imagem com as caixas desenhadas sera salva em:

```text
captures/result.jpg
```

Antes desse teste, confirme que o arquivo abaixo existe:

```text
models/yolov8n.onnx
```

Se existir apenas `yolov8n.pt`, o modelo ainda nao foi exportado para ONNX. Mesmo assim, `scripts/count_image.py` consegue testar a imagem usando `yolov8n.pt`.

## Se o main.py abrir e fechar sozinho

Rode pelo PowerShell para ver a mensagem:

```powershell
python main.py
```

Se ainda fechar, veja o arquivo:

```text
app_error.log
```

Esse arquivo e criado quando o app encontra erro ao iniciar.

## Rodar no Android

Voce tem tres caminhos praticos:

1. Gerar o APK pelo GitHub Actions, sem instalar Linux no seu PC.
2. Usar WSL no Windows.
3. Usar uma maquina Linux, uma VM ou um servidor.

Para quem esta no Windows e nao quer instalar Linux, o caminho mais simples e o GitHub Actions.

## Gerar APK sem Linux local

Este projeto ja inclui o arquivo:

```text
.github/workflows/build-android.yml
```

Passo a passo:

1. Crie um repositorio no GitHub.
2. Envie estes arquivos para o repositorio.
3. No GitHub, abra a aba **Actions**.
4. Selecione **Build Android APK**.
5. Clique em **Run workflow**.
6. Espere terminar.
7. Baixe o artefato chamado **contador-objetos-apk**.
8. Extraia o ZIP e instale o `.apk` no celular Android.

No celular, talvez seja necessario permitir instalacao de apps de fontes desconhecidas.

## Gerar APK com WSL/Linux

No Linux/WSL:

```bash
python3 -m pip install --user buildozer
buildozer android debug
```

Depois de compilar:

```bash
buildozer android deploy run
```

O arquivo `.apk` ficara em `bin/`.

## Testar direto no celular

O teste mais fiel e instalar o APK gerado pelo GitHub Actions ou Buildozer.

Tambem da para testar partes do codigo pelo Pydroid 3, mas camera + Kivy + OpenCV no Android podem exigir configuracao extra e nem sempre funcionam igual a um APK empacotado. Para este projeto, trate o Pydroid como teste de Python e o APK como teste real do app.

## Observacoes importantes

- O modelo padrao detecta classes COCO, como `person`, `bottle`, `cup`, `cell phone`, `book`, `chair`, etc.
- Para contar objetos especificos que nao existem no COCO, sera preciso treinar ou usar outro modelo.
- O arquivo `models/yolov8n.onnx` nao fica salvo no repositorio por ser um binario grande. Use `scripts/download_model.py` para baixar o YOLOv8n oficial e exportar para ONNX.
- Em celulares mais simples, use modelos pequenos como YOLOv8n para manter o desempenho aceitavel.
