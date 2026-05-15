# Klasyfikator Kształtów w PyTorch

Projekt ten to prosta i lekka sieć neuronowa (MLP) napisana w bibliotece PyTorch, która służy do klasyfikacji obrazów o rozdzielczości 8x8 pikseli. Model rozpoznaje cztery rodzaje kształtów: strzałkę w górę, w lewo, w prawo oraz znak "X".

Zamiast przekazywać całą macierz na wejście, projekt stosuje ekstrakcję cech poprzez projekcje X/Y, co redukuje dane z 64 pikseli do 16-elementowego wektora wejściowego. Dzięki temu sieć jest na tyle mała (architektura: 16 wejść -> 13 neuronów ukrytych -> 4 wyjścia), że mogłaby zostać łatwo zaimplementowana np. w układach FPGA.

## Struktura projektu

Wszystkie pliki muszą znajdować się w tym samym katalogu.

* **`gen_lib.py`** – Biblioteka generująca zbiory danych. Zawiera definicje wzorców, aplikuje przesunięcia (shift) oraz szum (noise), co pozwala na wygenerowanie różnorodnego zestawu testowego i treningowego.
* **`mlp_utils.py`** – Plik narzędziowy zawierający definicje modelu (klasa `ShapeClassifier`), algorytm ekstrakcji cech oraz funkcje pomocnicze do ładowania wag i predykcji.
* **`teach.py`** – Skrypt do trenowania modelu. Pobiera surowe obrazki z generatora, ekstrahuje cechy, przeprowadza proces uczenia i zapisuje wytrenowane wagi do pliku `model_strzalek.pth`.
* **`test.py`** – Skrypt służący do masowego testowania modelu na nowo wygenerowanym zestawie danych. Zwraca skuteczność klasyfikacji w procentach.
* **`run.py`** – Pozwala na manualne przetestowanie modelu na konkretnej, wpisanej w kodzie macierzy 8x8.
* **`mulit_test.py`** – Zaawansowany skrypt testujący, który symuluje odczyt 6 zaszumionych klatek, tworzy z nich zsumowaną "mega klatkę", filtruje ją i podejmuje ostateczną decyzję metodą większościowego głosowania (majority vote).

## Instalacja

1. Sklonuj repozytorium na swój dysk:
   ```bash
   git clone https://github.com/xorgzz/Arrow_Recognition_MLP
   cd Arrow_Recognition_MLP
   ```

2. Zainstaluj wymagane pakiety używając przygotowanego pliku requirements.txt:
  ```bash
  python3 -m pip install -r requirements.txt
  ``` 

## Instrukcja uruchomienia
Aby projekt działał poprawnie, musisz postępować w określonej kolejności, zaczynając od wygenerowania wag modelu.
### Krok 1: Wytrenuj model
Najpierw uruchom skrypt uczący, który stworzy zbiór danych i na jego podstawie wytrenuje sieć neuronową. Skrypt zapisze wynik do pliku model_strzalek.pth.
```bash
python3 teach.py
```
### Krok 2: Sprawdź celność na dużym zbiorze
Zweryfikuj skuteczność zapisanego modelu, uruchamiając masowe testy:
```bash
python3 test.py
```

### Krok 4: Test symulujący odczyt wielu klatek
Przetestuj odporność modelu w warunkach silnego zaszumienia wykorzystując algorytm analizy wielu klatek:
```bash
python3 mulit_test.py
```
