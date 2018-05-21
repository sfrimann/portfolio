# Harry Potter Textual Analysis

<img src="./hogwarts-crest-shield-emblem-logo-vector-school-of-witchcraft-and-wizardry-black-and-white.svg" width="50%" title="Never tickle a sleeping dragon.">

## Background
Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling, chronicling the life of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley. Since the publication of the first novel in 1997, more than 500 million copies have been sold making it the best-selling book series in history.

## Goals
* To familiarise myself with tools such as `nltk` and WikiData `SPARQL` queries
* Perform simple descriptive analyses, such as word count and vocabulary analysis
* Extract characters from the text for cool stuff like network analyses

## Data
The data used in the project are the seven Harry Potter e-books published by Scholastic. The e-books are in `.epub` format, which is a format that embeds HTML files in a zipped archive. The `.epub` files are read using the python `ebooklib` package (https://github.com/aerkalov/ebooklib).

For copyright reasons the actual e-books are not included in the repository.

## Notebooks
* [Harry Potter and the Preprocessing](./1_harry_potter_preprocessing.ipynb): Import of the books and preprocessing into tokens using `nltk`
* [Harry Potter and the Descriptive Analysis](./2_harry_potter_descriptive_analysis.ipynb): Various simple descriptive analyses
* [Harry Potter and the Friends](./3_harry_potter_friends.ipynb): Extract characters from text to see who appears most often and where