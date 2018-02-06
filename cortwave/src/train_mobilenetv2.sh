#!/bin/bash
for i in 0 1 2 3 4;
  do python3 model.py train --architecture=mobilenetv2 --fold=$i --lr=0.001 --batch-size=32 --iter-size=1 --epochs=20 --optim=adam --cached-part=1.0 --crop-central=True;
  python3 model.py train --architecture=mobilenetv2 --fold=$i --lr=0.001 --batch-size=16 --iter-size=1 --epochs=50 --optim=adam;
done
