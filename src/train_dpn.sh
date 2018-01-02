#!/bin/bash
for i in 0 1 2 3 4;
  do python3 model.py train --architecture=dpn68 --fold=$i --lr=0.001 --batch-size=60 --iter-size=4 --epochs=30 --optim=adam;
done