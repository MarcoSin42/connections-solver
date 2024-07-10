# Introduction

Connections is New York Times game where you are given 16 words.  Your goal is to group 4 words which have something in common, you do this 4 times since you have 16 words.  This program is an attempt to solve it aswell as presenting data to prove the validity of this model.

You can play [Connections here.](https://www.nytimes.com/games/connections)

# Usage
Use the solver script contained within the scripts file directory like so:

    python solver.py <Path to file containing 16 connections words> 

For example if you navigated to the scripts directory:

    python solver.py ../example

**Warning**: the model does not handle words contained outside of its dataset.

# How am I solving it?

I am using word embeddings to solve this, specifically Google's pretrained [Word2Vec](https://code.google.com/archive/p/word2vec/).

Some Google software developer makes the interesting observation that word vectors capture semantic similarity.  This is useful for solving our problem since the name of the game is group words with similiar meanings.


# More about NYT Connections

To understand why the model performs poorly on some words and better on others, we must first explain the game a bit more thoroughly.

The game has four levels of difficulty contained per game.  With increasing difficulties the association between words become more loose.  For example, for the level 0 group we may have a group of fruits like: banana, apple, orange, mango.  And for more difficult levels, for example, level 4, we'd have more loose associations such as the group "Letter Homophones" of which its members are: 'are', 'queue', 'sea', and 'why.'

# The Data

![plot](https://raw.githubusercontent.com/MarcoSin42/connections-solver/main/images/Validation_plot.png)


# Analysis and conclusion

As can be seen in the above plot, the model performs relatively well on the easier levels and performs more poorly on the more difficult levels.  This could be due to model chosen Word2Vec being trained on news articles as opposed to poetry or a more creative writing corpus, this is because news articles are meant to be interpretted more literally whereas the higher difficulty levels are increasingly abstract.  An improvement could be made by using a different model trained on more creative works.  
