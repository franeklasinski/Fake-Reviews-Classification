# Klasyfikacja Fałszywych Recenzji

Projekt machine learning do wykrywania fałszywych recenzji produktów przy użyciu przetwarzania języka naturalnego.

## Opis

Projekt implementuje 5 różnych podejść machine learning do klasyfikacji recenzji jako prawdziwe (OR) lub fałszywe (CG). Łączy analizę tekstu z inżynierią cech numerycznych, osiągając wysoką dokładność w wykrywaniu fake reviews.

## Główne Funkcje

- **Analiza tekstu**: Przetwarzanie NLP i ekstrakcja cech
- **5 modeli ML**: Porównanie różnych algorytmów
- **Model hybrydowy**: Połączenie cech tekstowych i numerycznych
- **Wizualizacje**: Szczegółowe wykresy i analizy
- **Analiza ważności cech**: Identyfikacja kluczowych słów

## Dataset

**kaggle** https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
Dataset zawiera:
- **Tekst recenzji** + **Etykiety** (CG=Fake, OR=Real)
- **Kategorie produktów** + **Oceny** (1-5 gwiazdek)
- **Podział**: 80% trening, 20% test

## Modele

### 1. Random Forest
- Cechy numeryczne (liczba słów, znaków, wykrzykników itp.)
- One-hot encoding kategorii

### 2. TF-IDF + Regresja Logistyczna  
- Wektoryzacja tekstu (5000 cech, n-gramy 1-2)
- Skalowanie macierzy rzadkich

### 3. Count Vectorizer + Naive Bayes
- Alternatywne podejście do wektoryzacji tekstu
- Multinomial Naive Bayes

### 4. Model Hybrydowy (Najlepszy)
- **Łączy cechy tekstowe + numeryczne**
- Custom transformer do fuzji cech
- Najwyższa dokładność

### 5. Gradient Boosting
- Ensemble method dla cech numerycznych
- Automatyczne wykrywanie interakcji między cechami

### Wnioski
- **Podejście hybrydowe najlepsze**: Kombinacja tekst + cechy numeryczne
- **Cechy tekstowe kluczowe**: Modele tekstowe przewyższają numeryczne
- **Inżynieria cech ma znaczenie**: Custom cechy poprawiają wyniki

## Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/franeklasinski/fake-reviews-classification.git
cd fake-reviews-classification

# Pobierz dane NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Główne Cechy Projektu

- **Analiza eksploracyjna**: Rozkład etykiet, kategorii, wzorce tekstowe
- **Inżynieria cech**: Liczba słów, znaków, zdań, wykrzykników, wielkich liter
- **Wizualizacje**: ROC curves, confusion matrix, ważność cech
- **Custom transformer**: Łączenie cech tekstowych i numerycznych
- **5 algorytmów ML**: Od Random Forest po modele hybrydowe

## Autor

**Franciszek Lasiński**
- GitHub: [@franeklasinski](https://github.com/franeklasinski)

---

