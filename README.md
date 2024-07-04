# Introduction

Connections is New York Times game where you are given 16 words.  Your goal is to group 4 words which have something in common, you do this 4 times since you have 16 words.

You can play [Connections here.](https://www.nytimes.com/games/connections)


# How am I solving it?

I am using word embeddings to solve this, specifically Google's pretrained [Word2Vec](https://code.google.com/archive/p/word2vec/).

Some Google software developer makes the interesting observation that word vectors capture semantic similarity.  This is useful for solving our problem since the name of the game is group words with similiar meanings.


