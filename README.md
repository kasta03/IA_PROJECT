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