# Mountain-Car
* This is for DSAI homework4
* Demo
* ![](https://imgur.com/VMlD1Q0.gif)

## Idea
* I implement a Deep-Q Network to make the car climb to the mountain.
* When the random action makes the mission success, I reward the whole sequence of previous actions to encourage the agent learn from it.
* See some analysis in `Report.pdf`

## Run
* Use exist model to perform the task.
    * `$ python3 MountainCar.py --train=False --modelPath=[the path to model]`
    * There are some pre-trained models under `/model/` directory.
    * After finish model training, it will show the accuracy of the model by run 500 rounds and calculate the number of rounds that the car reach the top flag.
    * In the end, the program will show the Mountain-Car window and play the game one time to let you see the result.
* Retrain the model by random data.
    * `$ python3 MountainCar.py `
* Add `--plot=True` if you want to plot some line graph for analysis.

## Requirements
* `pip install -r requirements.txt`
* keras==2.1.6
* numpy==1.14.1
