#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=alexnet --fold=$i --lr=0.001 --batch-size=40 --iter-size=1 --epochs=60 --optim=adam;
done
