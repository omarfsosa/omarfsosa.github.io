---
layout: single
title:  "A simple example of ordinal regression"
description: "Because I'm hyped about The Queen's Gambit"
date:   2020-11-25
mathjax: true
tags: [ordinal regression, stan, statistics, chess, elo]
---
I binged the whole thing (because, you know, that's what we do in the times of Covid) and I was quite pleased with it. Isn't it nice when you start to watch a TV show without any expectations and it ends up being one of the best things of the year? Anyway, once Beth Harmon was out of the way, I had time to think about some chess-stats questions. In particular there were 2 scenes in this show that left me thinking (SPOILERS AHEAD!):

* First, when Mr. Sheibel is starting to teach Beth how to play, he always plays the white pieces. So how much does playing white increases one's probability of winning?
* And second, when Beth goes to sign up for her first tournament but she does not have an Elo rating, the organizers give her a bit of a hard time. How good are Elo ratings anyway? Can I come up with a better rating?

No doubt there's been tons of research about these questions and I bet quick google search will suffice. But that's no fun. Instead, I've downladed a bunch of games from the [FIDE World Cups](https://theweekinchess.com/chessnews/events/fide-world-cup-2019) and I've decided to write my own model to answer my questions. Of course, FIDE world cups are far from a representative sample of an average chess game, so all results I'll show here only apply to FIDE world cup games (or perhaps other games outside the world cup but that involve players just as skilled and similar playing circumstances).

Here's the data
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

The first thing I want to try is just an Elo based model. That is, a model that tries to predict the game outcome based on the Elo of each player. But simply saying that the player with higher Elo would win is not a good idea. Such approach would miss the fact that playing white pieces _might_ give you an edge. For example, Elo ratings are meant to be such that the probability of winning for player A is given by something like $$P_w(A) = 1 / (1 + 10^{-(E_A - E_B)/400})$$, but this formula assumes that the game is "fair" and it also ignores (I think) the possibility of a draw. If we want to take into account the possibility of an edge for player A, the least we can do is model the probability of winning as something like

\begin{align}
P_{lose}(A) &= 1 - 1 / (1 + 10^{-(\alpha + E_A - E_B - c_1)/400}) \newline
P_{draw}(A) &=  1 / (1 + 10^{-(\alpha + E_A - E_B - c_1)/400}) -   1 / (1 + 10^{-(\alpha + E_A - E_B - c_2)/400})\newline
P_{win}(A) &= 1 / (1 + 10^{-(\alpha + E_A - E_B - c_2)/400}),
\end{align}
where $$\alpha$$ is a term representing first move advantage and the cutpoints $$c_1$$ and $$c_2$$ help me to account for the 3 possible outcomes. This is an **ordinal regression model** (ordinal regression because each game can end in Lose, *Draw* or Win), in which the _effective_ elo difference is modeled with a logistic distribution.

I'm going to fit this model in Stan (what else?). However, I'm not going to enforce the scale of my logistic distribution to be $$log(10)/400$$, I'll let my model learn the right scaling. If you are new to ordinal regression, I recommend you read [this post](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html) before you carry on.

The Stan code looks like this:
<div class="input_area" markdown="1">  

```c++
data {
    int n_games;
    int result_category[n_games]; // 1: Lose, 2: Draw, 3: Win
    real white_elo[n_games];
    real black_elo[n_games]; 
}
parameters {
    real<lower=0> gap;
    real<lower=0> white_advantage;
    real<lower=0> scale;
}

transformed parameters {
    ordered[2] c = to_vector({-gap, gap});
}

model {
    gap ~ exponential(1);
    white_advantage ~ exponential(1);
    scale ~ std_normal(); // Implicit half normal
    for (g in 1:n_games) {
        result_category[g] ~ ordered_logistic(white_advantage + scale * (white_elo[g] - black_elo[g]), c);
    }
}
```
</div>

The model is pretty simple, and perhaps the only remarkable aspect of it is the fact that I'm not using 2 independent parameters for the cutpoints but, instead, I make one the negative of the other. I do this because I want to enforce that any unbalance in the game is taken into account only by the first-move-advantage term.

I fit the model using PyStan and I see that all my chains have mixed nicely on my first attempt[^1]

<div class="input_area" markdown="1" style="font size: 10px">  

```
Inference for Stan model: anon_model_5f684676ce3c9ea3dde56ce4f1a2ba1e.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

                  mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
gap               1.31  6.5e-4   0.03   1.24   1.28   1.31   1.33   1.38   2867    1.0
white_advantage   0.36  7.8e-4   0.04   0.28   0.33   0.36   0.39   0.45   3102    1.0
scale           6.3e-3  7.2e-6 3.8e-4 5.6e-3 6.0e-3 6.3e-3 6.5e-3 7.0e-3   2708    1.0
c[1]             -1.31  6.5e-4   0.03  -1.38  -1.33  -1.31  -1.28  -1.24   2867    1.0
c[2]              1.31  6.5e-4   0.03   1.24   1.28   1.31   1.33   1.38   2867    1.0
lp__             -2005    0.03   1.25  -2008  -2006  -2005  -2004  -2004   2017    1.0
```
</div>

According to this model, if you're playing white pieces against someone of the same Elo rating on a FIDE world cup, then

\begin{align}
P_{lose}(A) &= \mathrm{logit}^{-1}(-1.31)  &\approx 28\% \newline
P_{draw}(A) &= \mathrm{logit}^{-1}(0.36 + 1.31) - \mathrm{logit}^{-1}(0.36 - 1.31)  &\approx 56\% \newline
P_{win}(A)  &=  1 - \mathrm{logit}^{-1}(0.36 + 1.31) &\approx 16\% 
\end{align}

And if chess were a fair game **without first move advantage** then we would have

\begin{align}
P(win) = P(lose) &= \mathrm{logit}^{-1}(- 1.31)    &\approx 21\% \newline
P(draw)          &= 2 * \mathrm{logit}^{-1}(-1.31) &\approx 58\%
\end{align}

This means that playing white pieces increases your chances of winning by 7%, and decreases your chances of losing by 5%.

[^1]: Maybe it was not my first attempt but you have no way of knowing that so...

## Model 2: Constant Ability model.

Inspired by [this example](https://statmodeling.stat.columbia.edu/2014/07/13/stan-analyzes-world-cup-data/) about a slightly more popular kind of world cup, I decided to write a model for estimating players abilities using Elo scores only as a prior. I have the problem however, that the Elo scores I have in my dataset have been evolving in time. Capturing that effect is more trouble than I'm in the mood for, though. So, I'm going to assume that player's abilities are constant in time. I knooooooow this is not the case in real life but whatever. That's my assumption for now, and I'll stick to it unless it proves to be a really bad idea (it won't). To validate this model, I'll leave out the games of the 2019 world cup as a hold out set.

I'll use a partial pooling model, in which the _true_ players abilities are sampled from a common normal distribution. I will also take their average Elo score as a prior, to inform which side of the distribution each player is. So the model is

\begin{align}
\theta_i &= \mu + b E_i + \sigma \eta_i \newline
y_{ij} &\sim \mathrm{OrderedLogistic}(\alpha + \theta_i - \theta_j \vert -c, +c) \newline
\eta_i &\sim \mathrm{Normal}(0, 1)
\end{align},
where 
* $$\theta_i$$ is the ability of player $$i$$;
* $$\mu$$ is the mean ability of all players in the world cup. But since I'll only be comparing differences in abilities this term will be dropped in the actual model;
* $$E_i$$ is the (standardized) Elo score of player $$i$$,
* $$b$$ is a parameter to measure how important Elo scores are in determining the ability of each player;
* $$\sigma$$ is the residual (how much of the player's ability is not explained by the other 2 terms);
* The term $$\eta_i$$ is there because I'm using a non-centered parametrisation;
* $$y_{ij}$$ is the outcome of the game betweeen players $$i$$ (white) and $$j$$ (black);
* $$\alpha$$ is the first move advantage;
* And the $$c_i$$ are the cutpoints.

And I'm going to need some priors for all of this. I've chosen,
\begin{align}
b &\sim \mathrm{Normal}(0, 1),\newline
\alpha &\sim \mathrm{Normal}(0, 1),\newline
\sigma &\sim \mathrm{Student}_7(0, 2),\newline
c &\sim \mathrm{Exponential}(1),\newline
\end{align}

Here the actual Stan code:
<div class="input_area" markdown="1">  

```c++
data {
    int n_games;
    int n_players;
    int result_category[n_games];
    real first_elo_score[n_players];
    int<lower=1, upper=n_players> black_id[n_games];
    int<lower=1, upper=n_players> white_id[n_games];
}

transformed data {
    real elo_zscore[n_players];
    real mean_elo = mean(first_elo_score);
    real sigma_elo = sd(first_elo_score);
    for (n in 1:n_players) {
        elo_zscore[n] = (first_elo_score[n] - mean_elo) / sigma_elo;
    }
}

parameters {
    real<lower=0> gap;
    real white_advantage;
    real elo_importance; 
    real<lower=0> sigma;
    real raw_ability[n_players];
}

transformed parameters {
    ordered[2] c = to_vector({-gap, gap});
    real ability[n_players];
    for (p in 1:n_players) {
        ability[p] = elo_importance * elo_zscore[p] + sigma * raw_ability[p];
    }
}

model {
    raw_ability ~ normal(0, 1);
    gap ~ exponential(1);
    sigma ~ student_t(7, 0, 2);  // Implicit Half Student
    white_advantage ~ normal(0, 1);
    elo_importance ~ normal(0, 1);
    for (g in 1:n_games) {
        result_category[g] ~ ordered_logistic(
            white_advantage + ability[white_id[g]] - ability[black_id[g]],
            c
        );
    }
}
```
</div>


