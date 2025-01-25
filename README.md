# Letter Recognition App

## Spis treści
1. [Opis projektu](#opis-projektu)
2. [Główne funkcjonalności](#główne-funkcjonalności)
3. [Struktura repozytorium](#struktura-repozytorium)
4. [Wymagania systemowe](#wymagania-systemowe)
5. [Instrukcja uruchomienia](#instrukcja-uruchomienia)
   - [Instalacja zależności](#instalacja-zależności)
   - [Szkolenie modelu (`train.py`)](#szkolenie-modelu-trainpy)
   - [Uruchomienie aplikacji graficznej (`main.py`)](#uruchomienie-aplikacji-graficznej-mainpy)
6. [Analiza kodu i zasady działania](#analiza-kodu-i-zasady-działania)
   - [Plik `train.py` – trening i zapisywanie modelu](#plik-trainpy--trening-i-zapisywanie-modelu)
     - [Model `ConvNet`](#model-convnet)
     - [Funkcja `train_model`](#funkcja-train_model)
     - [Funkcja `evaluate_model`](#funkcja-evaluate_model)
     - [Funkcje zapisu/odczytu modelu (`save_checkpoint` i `load_checkpoint`)](#funkcje-zapisuodczytu-modelu-save_checkpoint-i-load_checkpoint)
     - [Funkcja `load_model_for_inference`](#funkcja-load_model_for_inference)
     - [Funkcja `get_data_loaders`](#funkcja-get_data_loaders)
     - [Funkcja `main()` w `train.py`](#funkcja-main-w-trainpy)
   - [Plik `main.py` – aplikacja okienkowa (GUI)](#plik-mainpy--aplikacja-okienkowa-gui)
     - [Klasa `ModernDrawingApp`](#klasa-moderndrawingapp)
     - [Struktura okna i elementów interfejsu](#struktura-okna-i-elementów-interfejsu)
     - [Proces rozpoznawania liter (`recognize`)](#proces-rozpoznawania-liter-recognize)
     - [Funkcja `main()` w `main.py`](#funkcja-main-w-mainpy)
7. [Możliwe rozszerzenia](#możliwe-rozszerzenia)
8. [Autorzy / Kontakt](#autorzy--kontakt)

---

## Opis projektu
**Letter Recognition App** to aplikacja, która pozwala na rysowanie litery na wirtualnym płótnie (canvas) i rozpoznawanie tej litery przy pomocy wcześniej wytrenowanego modelu sieci neuronowej (CNN). Projekt opiera się na zbiorze danych [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), który zawiera obrazy liter w formacie 28x28 pikseli.

Dzięki temu projektowi:
- Możesz przeprowadzić trening własnego modelu do rozpoznawania liter.
- Następnie uruchomić aplikację okienkową, w której narysujesz literę myszką, a sieć spróbuje zgadnąć, jaki to znak.

Projekt zawiera dwa główne pliki:
- **`train.py`**: Kod odpowiedzialny za trenowanie sieci neuronowej, walidację oraz zapisywanie modelu.
- **`main.py`**: Aplikacja graficzna (GUI) napisana w Tkinter, która wczytuje wytrenowany model i pozwala użytkownikowi rysować litery, a następnie je rozpoznawać.

---

## Główne funkcjonalności
1. **Trening modelu** – Skrypt `train.py` pobiera zbiór danych EMNIST (split `letters`) i przeprowadza proces uczenia głębokiej sieci neuronowej (CNN) z wykorzystaniem PyTorch.
2. **Zapisywanie i wczytywanie modelu** – Model zapisuje się do pliku `letter_model.pth`. Gdy plik istnieje, skrypt pozwala wczytywać istniejące wagi (kontynuacja nauki) lub korzystać z nich w trybie inference (w `main.py`).
3. **Aplikacja okienkowa** – `main.py` to prosty interfejs graficzny, gdzie:
   - Możesz rysować literę na czarnym tle za pomocą myszki.
   - Zmieniasz grubość pędzla (suwak).
   - Klikasz przycisk "Recognize" i otrzymujesz wynik w postaci trzech najbardziej prawdopodobnych liter wraz z ich prawdopodobieństwami.
4. **Możliwość czyszczenia rysunku** – Dzięki przyciskowi "Clear" można zresetować pole do rysowania.
5. **Obsługa spolszczonego zbioru liter** – W oryginalnym zbiorze EMNIST liter jest 26 (a tak naprawdę 27 klas). W kodzie zdefiniowano mapowanie `i -> chr(i + 96)` odpowiadające literom od `a` do `z`.

---

## Struktura repozytorium
W repozytorium znajdują się przykładowo następujące pliki i foldery:

- **`data/`** – Folder, gdzie automatycznie zostaną pobrane dane EMNIST przy pierwszym uruchomieniu treningu.
- **`letter_model.pth`** – Zapisany model PyTorch (czyli “wagi” sieci neuronowej). Pojawi się po zakończeniu treningu.
- **`main.py`** – Kod tworzący interfejs graficzny do rozpoznawania liter w czasie rzeczywistym.
- **`train.py`** – Kod do trenowania modelu i zapisywania/odczytu wag.
- **`README.md`** – Plik, który właśnie czytasz. Opisuje projekt.

---

## Wymagania systemowe
- Python 3.7+ (zalecane 3.9 lub wyższa).
- Biblioteki Python:
  - `torch` (PyTorch)
  - `torchvision`
  - `tkinter` (w większości systemów macOS/Windows jest wbudowane w instalację Pythona; w Linuxie może być wymagany dodatkowy pakiet `python3-tk`).
  - `Pillow`
  - `tensorboard` (opcjonalnie, jeśli chcesz monitorować trening).
  - Inne biblioteki wymienione w kodzie (np. `argparse`, `datetime`, `numpy` – w razie potrzeby).

Jeśli masz plik `requirements.txt`, możesz zainstalować wszystkie zależności poprzez:
```bash
pip install -r requirements.txt
```
lub zainstalować ręcznie:
```bash
pip install torch torchvision pillow tensorboard

```

**Uwaga**: Jeśli posiadasz kartę graficzną NVIDIA i sterowniki CUDA, PyTorch może korzystać z GPU w celu przyspieszenia treningu. W przeciwnym razie wszystko zadziała na CPU (będzie to jedynie dłuższe).

---
### Instrukcja uruchomienia

#### Instalacja zależności
1. Upewnij się, że masz zainstalowanego Pythona w wersji co najmniej 3.7.
2. (Opcjonalnie) Aktywuj wirtualne środowisko (venv), aby zainstalować zależności lokalnie.
3. Zainstaluj wymagane biblioteki:
```bash
pip install torch torchvision pillow tensorboard
```
4. Upewnij się, że możesz importować wszystkie potrzebne pakiety w Pythonie (sprawdzisz, wpisując `python` i w interpretatorze `import torch`, itp.).


#### Szkolenie modelu (`train.py`)
1. Jeśli **nie posiadasz** pliku `letter_model.pth` (czyli model nie jest jeszcze wytrenowany), uruchom w terminalu:

    ```bash
    python train.py
    ```
   - Skrypt automatycznie pobierze zbiór danych EMNIST (split `letters`) do folderu `data/`.
   - Rozpocznie się proces trenowania modelu sieci neuronowej – może to trochę potrwać (zależnie od CPU/GPU).
   - Po zakończeniu treningu powstanie plik `letter_model.pth`.

2. (Opcjonalnie) Możesz dostosować hiperparametry przez argumenty w wierszu poleceń, np.:
    ```bash
    python train.py --epochs 15 --batch_size 64 --lr 0.001
    ```

   - `--epochs`: liczba epok treningu.
   - `--batch_size`: rozmiar paczki danych.
   - `--lr`: współczynnik uczenia (learning rate).

3. Jeśli **masz** już plik `letter_model.pth`, to skrypt `train.py` automatycznie załaduje stare wagi i domyślnie sprawdzi, czy chcesz kontynuować trening. W kodzie sprawdzane jest, czy plik istnieje. Jeśli tak – ładuje się model, jeżeli nie – następuje trenowanie od zera.

**Uruchomienie aplikacji graficznej (`main.py`)**

1. Po zakończeniu treningu (i posiadaniu pliku letter_model.pth) uruchom:
    ```bash
     python main.py
    ```
2. Otworzy się okienko aplikacji Tkinter:

    - Górny napis "Letter Recognition"
    - Pole rysowania (czarne tło 280x280 pikseli)
    - Przycisk "Clear" do czyszczenia płótna.
    - Przycisk "Recognize" do rozpoznania.
    - Suwak zmieniający grubość "pędzla".
    - Pole, w którym wyświetlą się wyniki rozpoznawania (top 3 przewidywania).

3. Sposób użycia:

   - Narysuj na płótnie jakąś literę (możesz klikać i przeciągać myszką, tak jakbyś rysował farbą).
   - Kliknij "Recognize".
   - W okienku "Most likely predictions:" pojawią się trzy najbardziej prawdopodobne litery wraz z wartościami `p = ...` (oznaczającymi pewność predykcji).

### Analiza kodu i zasady działania
Poniżej znajduje się bardziej szczegółowy opis najważniejszych elementów kodu.
**Plik `train.py` – trening i zapisywanie modelu**
**Model ConvNet**
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 27)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
- Klasa `ConvNet` dziedziczy po `nn.Module`, co oznacza, że jest to sieć neuronowa w PyTorch.
- Składa się z:
  - Dwóch warstw konwolucyjnych (`conv1`, `conv2`) z filtrami 3x3.
  - Normalizacji BatchNorm (`batch_norm1`, `batch_norm2`) po każdej konwolucji w celu stabilizacji i przyspieszenia treningu.
  - Warstw Dropout (`dropout1`, `dropout2`), które pomagają uniknąć przeuczenia (overfittingu).
  - Warstw w pełni połączonych (ang. fully connected, `fc1` i `fc2`).
- Ostatnia warstwa ma 27 neuronów wyjściowych (w EMNIST Letters bywa 26 lub 27 klas w zależności od konfiguracji; tutaj przyjęto 27).
- W metodzie `forward` wykonujemy operacje aktywacji ReLU, pooling (max_pool2d), flattenowanie i na końcu `log_softmax`, który jest często używany w PyTorch do zadań klasyfikacji.

**Funkcja `train_model`**
## Funkcja `train_model`
```python
def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0005, log_dir="logs"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        scheduler.step()
        val_acc = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch} | Avg Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}')

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()
    return model
```
