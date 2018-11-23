---
layout: post
title: "The Variational Renormalization Group and Deep Leaning: a Naive Attempt"
date: 2018-10-19 18:08:00 -0700
comments: false
---

# Boltzmann machines and the Ising model

As a physicist, what first got me interested in machine learning was hearing
about [Boltzmann machines](https://en.wikipedia.org/wiki/Boltzmann_machine), and
how training them was performing the *inverse* computation to what physicists
typically learn to do in statistical mechanics.

The workhorse of stat mech is the [Ising
Model](https://en.wikipedia.org/wiki/Ising_model), which in its most general
form is an $$N$$ node undirected graph with all-to-all connections. It's well
known for being a simple model of [how fucking magnets
work](https://www.youtube.com/watch?v=lFabsRFnWy0), with the simplest
application being to ferromagnetism.

(Statitician): *Well actually*, the classical Ising Model is a *Markov Random
Field*: a tuple $$(G, E, f)$$ where $$G$$ is an undirected graph, and $$E$$ a
set of edges, which together define a set of cliques $${C_v}$$ partitioning
$$G$$. $$X$$ are a set of random variables one-to-one associated with $$G$$, and
$$f$$ represents a potential function on the cliques of $$G$$.
