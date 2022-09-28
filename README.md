# Reinforcement Learning: policy evaluation and iteration
## Jack’s Car Rental (from Sutton & Barto's _Reinforcement Learning ─ An Introduction_)
A small project, from this [reference book](http://incompleteideas.net/book/RLbook2020.pdf) by Sutton & Barto, that I implemented when I encountered it. The goal of this exercise, in the chapter on **Dynamic Programming**, was to implement a _policy evaluation and iteration algorithm_, before later discussing value iteration. It is specifically designed to have a rather low dimension state space (~400), since classical DP requires to compute a complete model of the environment.

### Problem statement

Example **4.2** from the book:
> Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited <span>$</span>10 by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned. To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of <span>$</span>2 per car moved. We assume that the number of cars requested and returned at each location are Poisson random variables, meaning that the probability that the number is $n$ is 
$\frac{\lambda^n}{n!}e^{-\lambda}$, where $\lambda$ is the expected number. Suppose $\lambda$ is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be $\gamma = 0.9$ and formulate this as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight. Figure 4.2 shows the sequence of policies found by policy iteration starting from the policy that never moves any cars.

The authors provide us with the results for this example, which are displayed as follow:

![example-4 2-book](https://user-images.githubusercontent.com/114467748/192849100-19008b5f-b6b2-4f4b-9964-36cd0805c376.png)

### Exercise **4.7**

The exercise is the following:
> Write a program for policy iteration and re-solve Jack’s car rental problem with the following changes. One of Jack’s employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs <span>$</span>2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location. If more than 10 cars are kept overnight at a location (after any moving of cars), then an additional cost of <span>$</span>4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your program, first replicate the results given for the original problem.

The code from this repo respectively replicates _Example **4.2**_ (upper part) and solves _Exercise **4.7**_ (lower part):

![exercise-4 7-me](https://user-images.githubusercontent.com/114467748/192849833-ffa0c03c-d8a2-40be-adbd-b73e62cfaf10.png)

Without even taking into account the number of actions, despite the number of states $\mathcal{S}\simeq 400$ being quite low, computing the entire world's dynamics still requires $O(\mathcal{S}^3)$ operations and $O(\mathcal{S}^2)$ memory space. One can quickly realize what the _curse of dimensionality_ appears quite rapidly in RL!

### A bit more

Just for fun, I ran the simulation again with `max_cars_A = max_cars_B = 40`, while stating that **every** extra 10 parking spots (up to 40 max in total) still costs <span>$</span>4:

![max40cars-me](https://user-images.githubusercontent.com/114467748/192899942-b1ec5fce-1c28-44de-b134-22bb2b92e6fe.png)

Cheers!

