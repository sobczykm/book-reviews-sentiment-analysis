# Analiza Sentymentu Recenzji Książek

Dokumentacja projektu zaliczeniowego w ramach przedmiotu **Inżynieria Oprogramowania**

---

## Autorzy
- Julia Brauze
- Marcelina Momotko
- Mateusz Sobczyk

---

## Spis treści
1. [Charakterystyka oprogramowania](#charakterystyka-oprogramowania)
2. [Specyfikacja wymagań](#specyfikacja-wymagań)
   - [Wymagania funkcjonalne](#wymagania-funkcjonalne)
   - [Wymagania niefunkcjonalne](#wymagania-niefunkcjonalne)
3. [Architektura systemu](#architektura-systemu)
4. [Opis modelu analizy sentymentu](#opis-modelu-analizy-sentymentu)
5. [Interfejs użytkownika](#interfejs-użytkownika)
6. [Scenariusze testów](#scenariusze-testów)
7. [Dane i źródła](#dane-i-źródła)

---

## Charakterystyka oprogramowania

### Nazwa skrócona
**Book Sentiment Analysis**

### Nazwa pełna
System analizy sentymentu recenzji książek

### Krótki opis

Celem projektu jest stworzenie prostego systemu do **automatycznej analizy sentymentu tekstu** (recenzji książki) - ocenia, czy ma ona wydźwięk pozytywny czy negatywny (w skali 1–5).

Aplikacja składa się z:
- modelu klasyfikacji tekstu (analiza sentymentu),
- prostego interfejsu użytkownika (UI), w którym użytkownik może:
  - wpisać własną recenzję i otrzymać przewidywaną ocenę,
  - przeglądać statystyki dotyczące książek i recenzji

---

## Specyfikacja wymagań

### Wymagania funkcjonalne

| ID   | Nazwa                   | Opis                                                                                                     | Priorytet |
| ---- | ----------------------- | -------------------------------------------------------------------------------------------------------- | --------- |
| F-1  | Liczba książek          | System wyświetla całkowitą liczbę książek dostępnych w zbiorze danych                                    | 1         |
| F-2  | Średnia ocena           | System oblicza i prezentuje średnią ocenę wszystkich książek                                             | 1         |
| F-3  | Średnia liczba recenzji | System wyświetla średnią liczbę recenzji przypadającą na jedną książkę                                   | 2         |
| F-4  | Rozkład ocen            | System prezentuje wykres rozkładu liczby książek dla ocen 1–5                                            | 2         |
| F-5  | Rozkład liczby recenzji | System prezentuje wykres liczby książek w przedziałach liczby recenzji                                   | 2         |
| F-6  | Wprowadzanie recenzji   | Użytkownik może wpisać własny tekst recenzji książki                                                     | 1         |
| F-7  | Analiza sentymentu      | System analizuje sentyment wpisanej recenzji przy użyciu modelu ML                                       | 1         |
| F-8  | Predykcja oceny         | System zwraca przewidywaną ocenę recenzji w skali 1–5                                                    | 1         |
| F-9  | Wizualizacja oceny      | Wynik predykcji prezentowany jest w formie graficznej (gwiazdki)                                         | 2         |
| F-10 | Prawdopodobieństwa klas | System wyświetla prawdopodobieństwa dla każdej możliwej oceny (1–5)                                      | 3         |
| F-11 | Uruchamianie analizy    | Analiza sentymentu uruchamiana jest po kliknięciu przycisku                                              | 1         |
| F-12 | Filtr zakresu lat       | Użytkownik może określić zakres lat wydania książek za pomocą suwaka                                     | 1         |
| F-13 | Filtr oceny             | Użytkownik może ustawić zakres średniej oceny książek                                                    | 1         |
| F-14 | Filtr liczby recenzji   | Użytkownik może ustawić zakres liczby recenzji książek                                                   | 1         |
| F-15 | Filtr kategorii         | Użytkownik może wybrać kategorię książki lub opcję „Dowolne"                                             | 2         |
| F-16 | Liczba dopasowań        | System wyświetla liczbę książek spełniających wybrane kryteria                                           | 2         |
| F-17 | Losowa rekomendacja     | System losowo wybiera jedną książkę spośród pasujących wyników                                           | 1         |
| F-18 | Prezentacja książki     | System wyświetla szczegóły książki: tytuł, autorów, kategorię, opis, ocenę, liczbę recenzji, datę i link | 1         |

### Wymagania niefunkcjonalne

| ID | Kategoria | Opis | Priorytet |
|----|----------|------|-----------|
| NF-1 | Wydajność | Analiza sentymentu powinna odbywać się w czasie krótszym niż 2 sekundy | 2 |
| NF-2 | Użyteczność | Interfejs powinien być prosty i intuicyjny | 1 |
| NF-3 | Czytelność kodu | Kod powinien być zgodny z zasadami clean code | 1 |
| NF-4 | Skalowalność | Model powinien umożliwiać późniejszą rozbudowę | 2 |
| NF-5 | Open source | Projekt oparty o otwarte biblioteki | 1 |
| NF-6 | Przenośność | Aplikacja powinna uruchamiać się lokalnie na różnych systemach | 2 |

---

## Architektura systemu

### Architektura logiczna

System składa się z trzech głównych warstw:
1. **Warstwa danych** – pliki CSV z recenzjami i metadanymi książek,
2. **Warstwa logiki** – model analizy sentymentu oraz przetwarzanie danych,
3. **Warstwa prezentacji** – aplikacja webowa oparta o Streamlit.

### Architektura rozwoju

| Technologia | Przeznaczenie |
|------------|--------------|
| Python | Główny język programowania |
| Streamlit | Interfejs użytkownika |
| Pandas | Przetwarzanie danych |
| Plotly | Wizualizacja danych |
| HuggingFace Transformers | Model NLP (BERT – planowane) |
| scikit-learn | Wsparcie klasyfikacji i ewaluacji |

### Architektura uruchomieniowa

- Aplikacja uruchamiana lokalnie
- Brak dedykowanego backendu (logika w aplikacji)
- Dane wczytywane z plików CSV

---

## Procedura uruchomienia

### Wymagania wstępne

- Python 3.8 lub nowszy
- [uv](https://github.com/astral-sh/uv) - szybki menedżer pakietów Python
- utworzenie folderu data
- dane z [kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data), dodane w folderze data
  - zmiana nazwy z Books_data.csv na books.csv
  - zmiana nazwy z Books_ratings.csv na ratings.csv

### Instalacja uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Instalacja zależności

```bash
# Instalacja zależności projektu
uv sync

# Alternatywnie, jeśli chcesz zainstalować tylko zależności bez tworzenia środowiska wirtualnego
uv pip install -e .
```

### Uruchomienie aplikacji

```bash
# Generowanie modelu
uv run python train_model.py
```

```bash
# Uruchomienie aplikacji Streamlit
uv run streamlit run app.py
```

### Inne przydatne komendy uv

```bash
# Dodanie nowej zależności
uv add nazwa-pakietu

# Usunięcie zależności
uv remove nazwa-pakietu

# Aktualizacja wszystkich zależności
uv sync --upgrade

# Wyświetlenie listy zależności
uv pip list
```

---

## Opis modelu analizy sentymentu

### Cel modelu

Celem modelu jest **przypisanie recenzji książki do jednej z 5 klas sentymentu (1–5)** na podstawie samego tekstu.

### Podejście

- Analiza sentymentu w oparciu o **NLP**,
- Docelowo: fine-tuning pretrenowanego modelu **BERT** z warstwą klasyfikacyjną,
- Wynikiem działania modelu jest prawdopodobieństwo przynależności do każdej klasy.

### Etapy przetwarzania
1. Wstępna eksploracja danych,
2. Czyszczenie tekstu (tokenizacja, normalizacja),
3. Ekstrakcja cech (embeddingi BERT),
4. Klasyfikacja sentymentu.
---

## Interfejs użytkownika

Aplikacja składa się z trzech głównych widoków:

### Home
- statystyki książek i recenzji,
- histogram ocen,
- rozkład liczby recenzji.

### Score Match
- pole tekstowe do wpisania recenzji,
- przycisk uruchamiający analizę sentymentu,
- wizualna prezentacja wyniku (gwiazdki).

### Book Finder
- filtrowanie książek po roku, kategorii, ocenie i popularności,
- losowa rekomendacja książki,
- wyświetlenie podstawowych informacji o książce.

---

## Scenariusze testów

| ID   | Wymaganie | Kroki                                       | Oczekiwany rezultat                                     | Status   |
| ---- | --------- | ------------------------------------------- | ------------------------------------------------------- | -------- |
| T-1  | F-1       | Otworzyć widok Home                         | Wyświetlana jest liczba książek w zbiorze danych        | ZROBIONE |
| T-2  | F-2       | Otworzyć widok Home                         | Wyświetlana jest średnia ocena wszystkich książek       | ZROBIONE |
| T-3  | F-3       | Otworzyć widok Home                         | Wyświetlana jest średnia liczba recenzji                | ZROBIONE |
| T-4  | F-4       | Otworzyć widok Home                         | Wyświetlany jest wykres rozkładu ocen 1–5               | ZROBIONE |
| T-5  | F-5       | Otworzyć widok Home                         | Wyświetlany jest wykres rozkładu liczby recenzji        | ZROBIONE |
| T-6  | F-6       | Przejść do widoku Score Match               | Możliwe jest wpisanie tekstu recenzji                   | ZROBIONE |
| T-7  | F-7, F-11 | Wpisać recenzję i kliknąć przycisk analizy  | System analizuje sentyment tekstu                       | ZROBIONE |
| T-8  | F-8       | Wpisać recenzję i uruchomić analizę         | System zwraca ocenę w skali 1–5                         | ZROBIONE |
| T-9  | F-9       | Uruchomić analizę recenzji                  | Wynik prezentowany jest w formie gwiazdek               | ZROBIONE |
| T-10 | F-10      | Uruchomić analizę i rozwinąć szczegóły      | Wyświetlane są prawdopodobieństwa ocen 1–5              | ZROBIONE |
| T-11 | F-12      | Przejść do Book Finder i ustawić zakres lat | Wyniki ograniczone są do wybranego zakresu              | ZROBIONE |
| T-12 | F-13      | Ustawić zakres ocen książek                 | Zwrócone książki spełniają kryterium ocen               | ZROBIONE |
| T-13 | F-14      | Ustawić zakres liczby recenzji              | Zwrócone książki spełniają kryterium liczby recenzji    | ZROBIONE |
| T-14 | F-15      | Wybrać kategorię książki                    | Wyświetlane są tylko książki z danej kategorii          | ZROBIONE |
| T-15 | F-16      | Ustawić filtry w Book Finder                | Wyświetlana jest liczba dopasowanych książek            | ZROBIONE |
| T-16 | F-17      | Kliknąć „Find me a book”                    | System losowo wybiera jedną książkę                     | ZROBIONE |
| T-17 | F-18      | Wyświetlić wynik wyszukiwania               | Pokazane są szczegóły książki (tytuł, autor, opis itd.) | ZROBIONE |

---

## Dane i źródła

- Zbiór danych: recenzje książek z platformy Amazon (Kaggle)

---
