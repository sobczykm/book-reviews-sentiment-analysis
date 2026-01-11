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

| ID | Nazwa | Opis | Priorytet |
|----|-------|------|-----------|
| F-1 | Analiza sentymentu recenzji | System umożliwia ocenę sentymentu wpisanej przez użytkownika recenzji książki | 1 |
| F-2 | Skala ocen | System zwraca ocenę sentymentu w skali 1–5 | 1 |
| F-3 | Interfejs tekstowy | Użytkownik może wprowadzić własny tekst recenzji | 1 |
| F-4 | Wizualny wskaźnik wyniku | Wynik analizy prezentowany jest w formie graficznej (np. gwiazdki) | 2 |
| F-5 | Statystyki zbioru danych | Aplikacja wyświetla podstawowe statystyki dotyczące książek i ocen | 2 |
| F-6 | Wyszukiwanie książek | Możliwość filtrowania książek po roku, kategorii, ocenie i liczbie recenzji | 3 |
| F-7 | Losowa rekomendacja | System losowo wybiera książkę spełniającą kryteria wyszukiwania | 3 |
| F-8 | Filtrowanie recenzji | Możliwość odrzucenia recenzji niskiej jakości (np. chaotycznych) | 3 |

---

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

### Stan obecny

Aktualnie w aplikacji zastosowana jest **funkcja tymczasowa**, która symuluje działanie modelu. Docelowo zostanie ona zastąpiona właściwym modelem ML.

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

| ID | Wymaganie | Kroki | Oczekiwany rezultat |
|----|----------|-------|---------------------|
| T-1 | F-1, F3 | Wpisać recenzję i kliknąć analizę | System zwraca ocenę sentymentu |
| T-2 | F-2 | Wpisać różne teksty | Wynik mieści się w skali 1–5 |
| T-3 | F-4 | Uruchomić analizę | Wyświetlane są gwiazdki |
| T-4 | F-5 | Otworzyć ekran Home | Wyświetlane są statystyki |
| T-5 | F-6 | Ustawić filtry i wyszukać książkę | Zwrócona książka spełnia kryteria |

---

## Dane i źródła

- Zbiór danych: recenzje książek z platformy Amazon (Kaggle)

---
