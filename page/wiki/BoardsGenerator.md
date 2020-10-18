# Класс `BoradsGenerator`
Преднезначен для преобразования сырых ROOT-файлов в набор "досок" для последующей работы с нейронными сетями

## Пример работы с классом

```python
boardsGen = BoardsGenerator("mu-pi-_1500MeV_30.0k.root")
boardsGen.GenerateBoards(1000)
boardsGen.AddROOT("e-mu-_1600MeV_30.0k.root")
boardsGen.GenerateBoards(3000)
boardsGen.Save("data.csv")
boards, sizes = boardsGen.GetBoards()
```

## Описание класса

### `BoardsGenerator(*rootFiles)`

Конструктор класса `BoardsGenerator` 

#### Аргументы:
`rootFiles`, пути до ROOT-файлов с моделированием.
Эти ROOT-файлы будут использоваться для генерации досок.

`BoardsGenerator()`, конструкция без аргументов будет использована для генерации досок, состоящих только из шума.

#### Пример:

```python
boardsGen = BoardsGenerator("file1.root", "file2.root")
```
доски будут генерироваться на основе событий из двух ROOT-файлов (`"file1.root"`, `"file2.root"`)

### `BoardsGenerator.AddROOT(*rootFiles)`
Добавить ROOT-файлы для генерации досок

#### Аргументы:
`rootFiles`, пути до ROOT-файлов с моделированием.
Эти ROOT-файлы дополнительно будут использоваться для генерации досок.

#### Пример:
```python
boardsGen = BoardsGenerator("file1.root")
boardsGen.AddROOT("file3.root")
```
Доски будут генерироваться на основе событий из `"file1.root"`, `"file3.root"` файлов

### `BoardsGenerator.ClearROOT()`
Очистить набор файлов, по которым будут генерироваться доски

#### Пример:
```python
boardsGen = BoardsGenerator("file1.root")
boardsGen.ClearROOT()
boardsGen.AddROOT("file3.root")
```
Доски будут генерироваться на основе событий из `"file3.root"` файла

### `BoardsGenerator.GenerateBoards(n_boards, n_rings=1, size=(100, 100), freq=300, ticks=200, noise_level=100, augmentations=[Augmentator.Shift, Augmentator.Rotate, Augmentator.Rescale])`
Генерировать набор досок на основе событий из используемых ROOT-файлов

#### Аргументы:
* `n_boards`, количество досок, которое будет сгенерировано
* `n_rings`, количество сигнальных событий в генерируемых досках, по умолчанию 1
    * либо в формате кортежа `(минимальное количество колец на доске включительно, максимальное количество колец на доске включительно)`
    * либо число типа `int` = количество колец на каждой доске
* `size`, размеры досок по (`x`, `y`) осям в пикселях, по умолчанию (100, 100).
* `freq`, частота вычитки (в кГц), по умолчанию 300 кГц
* `ticks`, разрешение детектора по времени (в пкс), по умолчанию 200 пкс
* `noise_level`, уровень шума в детекторе (в кГц/мм^2), по умолчанию 100 кГц/мм^2
* `augmentations`, используемые аугментации для генерации досок, передаются в виде листа методов класса `Augmentator`. Подробнее об аугментациях [здесь](Augmentator.md)

#### Указания:
* Если параметры `size`, `freq` или `ticks` изменятся при повторном вызове метода, то будет выбрасываться ошибка.

#### Пример:
```python
boardsGen = BoardsGenerator("file1.root")
boardsGen.GenerateBoards(1000)
```
На основе `file1.root` будет сгенерированно 1000 досок с параметрами по умолчанию

### `BoardsGenerator.SaveBoards(filepath)`
Записать сгенерированные доски в файл `filepath`, где `filepath` - json файл с полями `boards` и `sizes`

#### Аргументы:
* `filepath`, путь до файла, в который будут сохранены доски

#### Пример:
```python
boardsGen = BoardsGenerator("file1.root")
boardsGen.GenerateBoards(1000)
boardsGen.SaveBoards("saveBoards.json")
```
Сгенерировать 1000 досок и записать их в файл `saveBoards.json` с полями `boards` и `sizes`

### `BoardsGenerator.GetBoards()`
Вернуть сгенерированные доски

#### Возвращает:
`(boards, sizes)`, кортеж, состоящий из `boards` 
* `boards`: `np.array` с размером `(N, 5)`, в столбцах которого лежат `x`, `y`, `t`, `s`, `id` зажжённых пикселей
    * `x`, `y`, координаты зажжённых пикселей (`np.int`)
    * `t`, время срабатывания пиксела (`np.int`)
    * `s`, пиксель является сигнальным или шумовым (`np.int`), `1`: сигнал, `0`: шум
    * `id`, идентификационный номер доски, к которой относится событие
* `sizes`: кортеж, состоящий из трёх `int` чисел, показывающий размеры доски по осям `x`, `y`, `t`.


#### Пример:
```python
boardsGen = BoardsGenerator("file1.root")
boardsGen.GenerateBoards(1000)
boards, sizes = boardsGen.GetBoards()
```
Сгенерировать 1000 досок и получить их

[**Вернуться на главную**](Home.md)
