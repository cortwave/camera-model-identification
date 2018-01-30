#!/bin/bash
for i in 0 1 2 3 4;
  do python3 model.py train --architecture=mobilenetv2 --fold=$i --lr=0.001 --batch-size=128 --iter-size=1 --epochs=100 --optim=adam;
done