---
layout: post
title: "Function Testing"
date: 2018-01-12 16:25:06 -0700
comments: false
---

We need to figure out how to do all the standard Mathematica/iPython notebook
embedding, along with $$ \LaTeX $$ rendering, proper rendering in Markdown by
the browser, etc..

We'll document all of that here, so this file will be continuously growing.

# Demo of basic functionality within markdown files for Jekyll blogging.

## Basics
Here's a [link](https://www.google.com) to google.

Here's some `verbatim code comment about an int x = 3`.


## Basic LaTeX
Let's test some inline math, like the basic $$ y = a x^2 + b $$. (NB the
alignment issues for $$ y = a x^2 + b $$ depending on browser zoom size..)

Here is the same math in its own line:

$$ y = a x^2 + b $$

We can also use normal $$\LaTeX$$ environments (TODO: explore other ones).

\begin{align}
    y =& a_1 x^2 + b_1 \newline
    z =& a_2 x^2 + b_2
\end{align}


## Basic code formatting
Here is some markdown stylee python code (NB it uses the same font, and has a
**white** background.

```python
s = "Python syntax highlighting"
print s
```

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

## iPython/Jupyter notebooks

A friend of mine claimed on Facebook:

> It was recently suggested that the NBA finals were rigged, perhaps to increase
> television ratings. So I did a simple analysis - suppose each game is a coin
> flip and each team has a 50% chance of winning each game. What is the expected
> distribution for the lengths the finals will go and how does it compare with
> reality?

> A simple calculation reveals P(4) = 8/64, P(5) = 16/64, P(6) = 20/64 and P(7)
> = 20/64.  How does this compare with history? Out of 67 series n(4) = 8, n(5)
> = 16, n(6) = 24 and n(7) = 19 so pretty damn close to the shitty coin flip
> model.

> TL:DR - a simple statistical model suggests the nba finals are not rigged

3 things:

1. I know nothing about basketball.
2. I don't think that anybody's rigging games.
3. Let's examine this claim closer.


```python
%matplotlib inline

# Standard imports.
import numpy as np
import pylab
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Resize plots.
pylab.rcParams['figure.figsize'] = 8, 4
```


```python
# Simulate 1000 series
game_lengths = []
for i in range(10000):
    wins_a = 0
    wins_b = 0
    for j in range(7):
        winning_team = np.random.rand() > .5
        if winning_team:
            wins_b += 1
        else:
            wins_a += 1

        if wins_a >= 4 or wins_b >= 4:
            break
    game_lengths.append(j + 1)
    continue

game_lengths = np.array(game_lengths)
plt.hist(game_lengths)
_ = plt.title('Game lengths under null hypothesis')
plt.xlabel('Game lengths')
```




    <matplotlib.text.Text at 0x7f91d84b8cd0>




![png](/NBA_files/NBA_2_1.png)



```python
print game_lengths.mean()
```

    5.8202


Indeed, the coin flip model ptestredicts that the distribution of game weights will have a lot of bulk around 6 and 7 games. What about historical games?


```python
game_lengths_historical = np.hstack(([4] * 8, [5] * 16, [6] * 24, [7] * 19))
plt.hist(game_lengths_historical)
_ = plt.title('Historical game lengths')
```


![png](/NBA_files/NBA_5_0.png)



```python
print game_lengths_historical.mean()
```

    5.80597014925


In fact, the historical game distribution indicates that the playoffs are slightly shorter than expected by chance. Does that mean that historical games haven't been rigged? Well, for one thing, you have to think about might cause game lengths to be shorter: blowouts. If one team is way better than the other, you would expect the series to be shorter than 7. In fact, the simulation with p = .5 represents the most extreme scenario, where every final is between two teams of exactly equal ability.

That is almost certainly never the case; consider the Boston Celtics winning streak of the 60's - they must have been much stronger than any other team! We can estimate the (implied) probability of winning from sports betting data. Sports betters have every incentive to produce calibrated predictions, because it directly impacts their bottom line.

I looked at the moneylines from 2004 - 2015:

http://www.oddsshark.com/nba/nba-finals-historical-series-odds-list
