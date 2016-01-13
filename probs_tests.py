import random
from numpy import log, exp, isnan, isinf, ceil

probs = {'a' : 0.5,
         'b' : 0.25,
         'c' : 0.1,
         'd' : 0.1,
         'e' : 0.05}

counts = {'a' : 20,
          'b' : 10,
          'c' : 15,
          'd' : 15,
          'e' : 25}

# linear space
linp = 1
for w in counts:
    for n in range(counts[w]):
        linp *= probs[w]

logp = log(1)
for w in counts:
    for n in range(counts[w]):
        logp += log(probs[w])

logp_mult = log(1)
for w in counts:
    logp_mult += (log(probs[w]) * counts[w])

logplog = log(1)
for w in counts:
    logplog += pow(log(probs[w]), log(counts[w]))
