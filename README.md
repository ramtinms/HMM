# HMM
This package provide HMM functionality like Viterbi, Forward-backward, Supervised learning params, Unsupervised learning params (EM)


## Parameters setting
You load parameters from a file with this format :
```
states (comma seperated)
\n
Pi Probabilities (comma seperated)
\n
transition Probabilities for source state 0 (comma seperated)
transition Probabilities for source state 1 (comma seperated)
...
transition Probabilities for source state N (comma seperated)
\n
emission probability for state 0 (comma seperated)
emission probability for state 1 (comma seperated)
emission probability for state 2 (comma seperated)
...
emission probability for state N (comma seperated)
```

for example see test.param

## Files
`sequence` stores a sequence of `observations`.



## Unit test
For testing functionality run `python test.py`
