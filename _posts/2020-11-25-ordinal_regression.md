---
title:  "A simple example of ordinal regression"
date:   2020-11-25
mathjax: true
---
The Queen's Gambit came out and I binged the whole thing because, you know, that's what we do in the times of Covid. I really enjoyed it, actually, but once Beth Harmon was out of the way, the chess-stats questions started to flow. In particular, there were 2 scenes in this show that left me thinking (SPOILERS AHEAD!):

* First, when Mr. Sheibel is starting to teach Beth how to play, he always plays the white pieces. It's only after a few games that he tells Beth that she can play white pieces too. So how much does playing white increases one's probability of winning?
* And second, when Beth goes to sign up for her first tournament, the organisers give her a bit of a hard time because she does not have an Elo rating. So, how do Elo ratings determine someone's probability of winning?

No doubt there's been tons of research about these questions already, and I bet quick google search will suffice to get an answer. But that's no fun. Instead, I've downloaded a bunch of games from the [FIDE World Cups](https://theweekinchess.com/chessnews/events/fide-world-cup-2019) and I've decided to write my own model to answer my questions. Of course, FIDE world cups are far from a representative sample of an average chess game, so all results I'll show here only apply to FIDE world cup games (or perhaps other games outside the world cup but that involve players just as skilled and similar playing circumstances).

After dealing with some formatting issues, here's the data in the always-friendly csv format:

<script src="https://gist.github.com/omarfsosa/f8753ac3a5199dd5205a062038c1daf5.js?file=blog__ordinal_regression__01.py"></script>


<div markdown="0" style="text-align: right">
    <table border="0" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th>date</th>
        <th>white_id</th>
        <th>black_id</th>
        <th>white_elo</th>
        <th>black_elo</th>
        <th>result</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>0</th>
        <td>2019-09-10</td>
        <td>9100075</td>
        <td>8603677</td>
        <td>1954</td>
        <td>2811</td>
        <td>0-1</td>
        </tr>
        <tr>
        <th>1</th>
        <td>2019-09-10</td>
        <td>24116068</td>
        <td>10207791</td>
        <td>2780</td>
        <td>2250</td>
        <td>1-0</td>
        </tr>
        <tr>
        <th>2</th>
        <td>2019-09-10</td>
        <td>8504580</td>
        <td>623539</td>
        <td>2284</td>
        <td>2774</td>
        <td>0-1</td>
        </tr>
        <tr>
        <th>3</th>
        <td>2019-09-10</td>
        <td>5202213</td>
        <td>6501311</td>
        <td>2767</td>
        <td>2387</td>
        <td>1-0</td>
        </tr>
        <tr>
        <th>4</th>
        <td>2019-09-10</td>
        <td>4902980</td>
        <td>4168119</td>
        <td>2407</td>
        <td>2776</td>
        <td>0-1</td>
        </tr>
        <tr>
        <th>...</th>
        </tr>
    </tbody>
    </table>
</div>


## Elo Based model

To answer my questions, I will fit a model that tries to predict the outcome of the game based on the Elo difference between the players. Of course, simply saying that the player with higher Elo would win is not good enough. Such approach would miss the fact that playing white pieces _might_ give you an edge. For example, in [this post](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/), FiveThirtyEight explain how they use an Elo based model for predicting the outcome of NFL games (a sport very similar to chess). Their model is such that:
\begin{align}
P_{win} = \frac{1}{1 + 10^{\frac{\mathrm{-EloDiff}}{400}}} \label{eq:elo} \tag{1}
\end{align}
As it is, this model assumes that the game is "fair", having both teams playing under equal circumstances. FiveThirtyEight then makes adjustments to the Elo of each team depending on other factors like if the team is playing at home or not, etc. Now, I cannot simply use this model because

1. I don't happen to be a chess expert unfortunately, so I don't know by how much I should change the Elo of players if they have white pieces. This is precisely my first question.
2. Equation $$\eqref{eq:elo}$$ tells me the Elo difference is being modelled as a logistic distribution (because the CDF is a logistic function). I don't know why this system has a problem with the Normal distribution, so I'll use a logistic distribution too just to be on the safe side. However, I doubt that $$\log(10)/400$$ would also be the right scale for Chess. So I will need to learn this from my model.
3. And finally, it'd seem to me that the above model would predict a tie only when both teams have the same Elo score. Ties are way more frequent in FIDE games than they are in NFL (I guess not that similar after all), so to account for this I need to turn the logistic regression model into an **ordinal regression** model.

That leaves me with the following model:

\begin{align}
P_{lose}(A) &= 1 - 1 / (1 + 10^{-(\alpha + E_A - E_B - c_1)/\sigma}) \newline
P_{draw}(A) &=  1 / (1 + 10^{-(\alpha + E_A - E_B - c_1)/\sigma}) -   1 / (1 + 10^{-(\alpha + E_A - E_B - c_2)/\sigma})\newline
P_{win}(A) &= 1 / (1 + 10^{-(\alpha + E_A - E_B - c_2)/\sigma})\label{eq:elo2} \tag{2},
\end{align}
where $$\alpha$$ is a term representing first move advantage and the cutpoints $$c_1$$ and $$c_2$$ help me to account for the 3 possible outcomes.

## Ordinal regression in Stan
I'm going to fit this model in Stan (what else?). The model code is quite simple and it looks like this:

<script src="https://gist.github.com/omarfsosa/f8753ac3a5199dd5205a062038c1daf5.js?file=blog__ordinal_regression__02.stan"></script>

Perhaps the only remarkable aspect of it is the fact that I'm not using 2 independent parameters for the cutpoints but, instead, I make one the negative of the other. I do this because I want to enforce that any unbalance in the game is taken into account only by the first-move-advantage term.

To build the model, I first put the data in the right shape and then pass it to Stan. I will also use ArViz to visualize my chains:

<script src="https://gist.github.com/omarfsosa/f8753ac3a5199dd5205a062038c1daf5.js?file=blog__ordinal_regression__03.py"></script>

Looking at the inference data I see that all my chains have mixed nicely on my first attempt[^1]

![Trace plot](assets/static/images/blog-images/2020-11-30-chess/traceplot.png)

And the mean estimates of my parameters are:

<div markdown="0" style="text-align: right">
    <table border="0" class="dataframe">
    <thead>
        <tr style="text-align: center;">
        <th>parameter</th>
        <th>mean</th>
        <th>std</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>white_advantage</th>
        <td>0.366611</td>
        <td>0.044193</td>
        </tr>
        <tr>
        <th>scale</th>
        <td>0.006281</td>
        <td>0.000375</td>
        </tr>
        <tr>
        <th>gap</th>
        <td>1.307304</td>
        <td>0.035150</td>
        </tr>
    </tbody>
    </table>
</div>


So, according to this model, if you're playing white pieces against someone of the same Elo rating on a FIDE world cup, then

\begin{align}
P_{win}(A) &= \mathrm{logit}^{-1}(0.36 - 1.31)  &\approx 28\% \newline
P_{draw}(A) &= \mathrm{logit}^{-1}(0.36 + 1.31) - \mathrm{logit}^{-1}(0.36 - 1.31)  &\approx 56\% \newline
P_{lose}(A)  &=  1 - \mathrm{logit}^{-1}(0.36 + 1.31) &\approx 16\% 
\end{align}

And if chess were a fair game **without first move advantage** then we would have

\begin{align}
P(win) = P(lose) &= \mathrm{logit}^{-1}(- 1.31)    &\approx 21\% \newline
P(draw)          &= 2 * \mathrm{logit}^{-1}(-1.31) &\approx 58\%
\end{align}

This means that **playing white pieces increases your chances of winning by 7%**, and decreases your chances of losing by 5%. To answer my second question I just need to plot equations $$\eqref{eq:elo2}$$ with the mean estimates. Here are my plots including the advantage for playing white pieces:

![Elo curves](assets/static/images/blog-images/2020-11-30-chess/elo_curves.png)


These results look sensible to me given my very limited experience in chess world cups, so I'm going to call it a day. If you'd like to learn more about ordinal regression, I recommend you read [this post](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html) by Mike Betancourt.


[^1]: Maybe it was not my first attempt but you have no way of knowing that so...


