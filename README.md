## Classificador de Imagens Biomédicas (DICOM)

Este projeto executa uma avaliação leave-one-out com 30 imagens (15 por classe):
- Classe positiva: `Pneumothorax`
- Classe negativa: `No Pneumothorax`

Para cada imagem de teste, o programa compara com as outras 29 e calcula, para cada método OpenCV (`CV_COMP_CORREL`, `CV_COMP_CHISQR`, `CV_COMP_INTERSECT`, `CV_COMP_BHATTACHARYYA`):
- Matriz de confusão (`TP`, `TN`, `FP`, `FN`)
- Sensibilidade
- Especificidade

## Pré-requisitos
- Python 3.10+
- `pip`

## Estrutura esperada
Mantenha os arquivos abaixo na raiz do projeto:
- `main.py`
- `requirements.txt`
- `Pneumothorax_files.txt`
- `No_Pneumothorax_files.txt`
- Pasta `train/` com as imagens DICOM

## Importante: caminho parcial nos arquivos `.txt`
Os arquivos `Pneumothorax_files.txt` e `No_Pneumothorax_files.txt` devem conter caminhos **parciais (relativos)**, e nao caminho absoluto.

Exemplo correto:
```txt
train/Pneumothorax/000001.dcm
train/No Pneumothorax/000000.dcm
```

Exemplo incorreto:
```txt
/home/usuario/projeto/train/Pneumothorax/000001.dcm
```

## Como executar

1. Crie um ambiente virtual:
```sh
python3 -m venv .venv
```

2. Ative o ambiente virtual:
- Linux/Mac:
```sh
source .venv/bin/activate
```
- Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

3. Instale as dependencias:
```sh
pip install -r requirements.txt
```

4. Execute o programa:
```sh
python main.py
```

## Saida esperada
O terminal exibira os resultados de cada metodo, incluindo:
- Metodo utilizado
- Estrategia/configuracao usada
- `TP`, `TN`, `FP`, `FN`
- Sensibilidade
- Especificidade

