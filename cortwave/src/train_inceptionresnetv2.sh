#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=inceptionresnetv2 --fold=$i --lr=0.001 --batch-size=4 --iter-size=1 --epochs=100 --optim=adam;
done
