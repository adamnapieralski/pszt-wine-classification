# Klasyfikacja win

## Dane

Dane wejściowe znajdują się w folderze data. Przygotowano również skrypt, który pobiera te dane z internetu. Można go uruchomić poleceniem:
```
python download_data.py
```

## Zależności

algorytmy:
    numpy
    sklearn - do testów

notebook:
    jupyter notebook
    pandas

## Instrukcja 

Algorytm Gradient Boosting oraz wykorzystywane przez niego drzewo decyzyjne znajdują się w module gboost. Interfejs obu modeli jest oparty na Scikit-learn.
```
import gboost
gb = gboost.GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, verbosity=0)
gb.fit(X_train, Y_train)
Y_pred = gb.predict(X_test)
score = gb.score(X_test, Y_test)
```

Moduł utilities zawiera funkcję do k-krotnej walidacji krzyżowej.

## Jupyter Notebook

W notebooku gradient_boosting_comparision znajduje się porównanie algorytmu gradient boosting z gboost oraz Scikit-learn.
