#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=densenet121 --fold=$i --lr=0.001 --batch-size=40 --iter-size=4 --epochs=30 --optim=adam;
done
